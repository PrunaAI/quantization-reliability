import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

import wandb
from src.config import RESULTS_DIR
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.common.dataset_result import DatasetResult
from src.dataset_processing.dataset_factory import DatasetFactory
from src.reliability_eval.common.score_types import DatasetMetricTypes, QuestionAggregateTypes
from src.reliability_eval.common.scores import QuestionAggregateScores
from src.reliability_eval.pipeline.evaluation_pipelines.accessors import ACCURACY_ACCESS_CONFIGS, SCORE_ACCESS_CONFIGS, TOKEN_ID_ACCESS_CONFIGS
from src.reliability_eval.pipeline.evaluation_pipelines.registry import EVALUATION_PIPELINES_DICT
from src.reliability_eval.pipeline.evaluation_pipelines.types import PipelineType
from src.reliability_eval.evaluator.accessor import ScoreAccessor
from src.reliability_eval.pipeline.calculator.metrics import MetricsCalculator
from src.reliability_eval.pipeline.config import BatchConfig, IntegratedPipelineConfig
from src.reliability_eval.pipeline.context import EvaluationContext, MetricsConfig
from src.reliability_eval.pipeline.processor.batch import DataLoader
from src.reliability_eval.pipeline.processor.grouper import _group_scores_by_perturbations
from src.reliability_eval.pipeline.processor.merger import ScoreMerger
from src.reliability_eval.pipeline.processor.processor import BatchProcessor
from src.reliability_eval.pipeline.results import EvaluationResults, GroupedResults
from src.reliability_eval.utils.model_loading import load_model_for_evaluation


logger = logging.getLogger("reliability_eval")

class IntegratedEvaluationPipeline:
    """Manages the complete evaluation pipeline."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.metrics_calculator = MetricsCalculator(
            MetricsConfig(metric_types=DatasetMetricTypes, round_accuracy=True)
        )

    def run_evaluation(
        self,
        config: IntegratedPipelineConfig
    ) -> EvaluationResults:
        """Runs complete evaluation pipeline."""
        logger.info("Starting evaluation pipeline")
        try:
            dataset = self._prepare_dataset(config)
            dataloader = self._prepare_dataloader(dataset, config.batch_config)
            evaluation_context = self._initialize_context(config)
            scores = self._process_batches(dataloader, evaluation_context)
            
            # Extract perturbation information if available
            if hasattr(config.dataset_config, 'source_type') and config.dataset_config.source_type == DatasetSourceType.MERGED:
                pert_types = [entry.metadata['perturbation_type'] for entry in dataset.entries]
                pert_intensities = [entry.metadata['perturbation_intensity'] for entry in dataset.entries]
                
                # Compute overall results
                overall_results = self._compute_final_results(scores, config)
                
                # Group scores and compute results for each group
                grouped_scores = _group_scores_by_perturbations(scores, pert_types, pert_intensities)
                grouped_results = {
                    group: self._compute_final_results(group_scores, config)
                    for group, group_scores in grouped_scores.items()
                }
                
                results = GroupedResults(
                    overall_results=overall_results,
                    grouped_results=grouped_results
                )
            else:
                results = self._compute_final_results(scores, config)
            
            # Generate Excel report with grouped results
            self._generate_evaluation_excel(
                results=results,
                config=config,
                dataset_entries=dataset.entries,
                num_max_entries=config.num_excel_entries
            )
            
            logger.info("Evaluation pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
            
    def _generate_evaluation_excel(
        self,
        results: Union[EvaluationResults, GroupedResults],
        config: IntegratedPipelineConfig,
        dataset_entries: List[DatasetEntry],
        num_max_entries: Optional[int] = None
    ) -> None:
        """Generates detailed Excel report of evaluation results."""
        import pandas as pd
        from pathlib import Path
        import os
        from datetime import datetime

        logger.info("Generating evaluation Excel report")
        
        # Create results directory structure
        base_path = Path(RESULTS_DIR)
        timestamp = datetime.now().strftime("%Y%m%d")
        exp_id = config.exp_id
        model_name = str(config.model_identifier).replace("/", "_")
        dataset_type = str(config.dataset_config.dataset_type.value)
        perturbation_type = str(config.dataset_config.perturbation_type.value) if hasattr(config.dataset_config, 'perturbation_type') else "none"
        perturbation_intensity = str(config.dataset_config.perturbation_intensity) if hasattr(config.dataset_config, 'perturbation_intensity') else "none"
        num_entries = len(dataset_entries)
        num_repeats = config.generation_config.num_repeats
        temperature = config.generation_config.temperature
        max_new_tokens = config.generation_config.max_new_tokens
        random_seed = config.random_seed
        config_string = f"exp-{exp_id}_model-{model_name}_dataset-{dataset_type}_pert-{perturbation_type}_int-{perturbation_intensity}_entries-{num_entries}_repeats-{num_repeats}_temp-{temperature}_tokens-{max_new_tokens}_seed-{random_seed}_time-{timestamp}"
        result_dir = base_path / config_string
        os.makedirs(result_dir, exist_ok=True)
        excel_path = result_dir / f"results_{config_string}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write all results to a single sheet
            self._write_results_to_excel(
                writer=writer,
                results=results,
                config=config,
                dataset_entries=dataset_entries,
                sheet_name='Results',
                num_max_entries=num_max_entries
            )
            
        logger.info(f"Saved evaluation results to {excel_path}")
    
    def _build_row(
        self,
        idx: int,
        entry: DatasetEntry,
        config: IntegratedPipelineConfig,
        pert_type: Optional[str],
        pert_intensity: Optional[int]
    ) -> Dict:
        """Builds the base row dict shared by grouped and ungrouped Excel export."""
        return {
            'entry_idx': idx,
            'exp_id': config.exp_id,
            'question': entry.question,
            'answer': entry.answer,
            'dataset_name': config.dataset_config.dataset_name,
            'perturbation_type': pert_type,
            'perturbation_intensity': pert_intensity,
            'model': str(config.model_identifier),
            'device': config.device,
            'seed': config.random_seed,
            'generation_strategy': config.generation_config.generation_strategy.value,
            'prompt_strategy': config.generation_config.prompt_strategy.value,
            'num_repeats': config.generation_config.num_repeats,
            'max_new_tokens': config.generation_config.max_new_tokens,
            'temperature': config.generation_config.temperature,
        }

    def _populate_pipeline_scores(self, row: Dict, idx: int, scores: Dict) -> None:
        """Adds per-pipeline scores (and accuracy, if present) to a row in place."""
        for pipeline_name, score_tensor in scores.items():
            pipeline_name = pipeline_name.value if isinstance(pipeline_name, PipelineType) else pipeline_name
            try:
                if idx >= len(score_tensor):
                    continue
                if pipeline_name == 'token_info':
                    token_info = score_tensor[idx]
                    row['token_info_tuples'] = str(token_info)
                    try:
                        row['first_token_probability'] = token_info[0][0][2]
                    except (IndexError, TypeError):
                        row['first_token_probability'] = None
                elif pipeline_name == 'full_text':
                    row['full_decoded_text'] = str(score_tensor[idx])
                elif pipeline_name == 'effective_lengths':
                    row['effective_lengths'] = str(score_tensor[idx])
                elif pipeline_name == 'padded_sequences':
                    row['padded_sequences'] = str(score_tensor[idx])
                elif pipeline_name == 'semantic_entropy':
                    row['semantic_entropy'] = str(score_tensor[idx])
                elif pipeline_name != "accuracy":
                    row[f'{pipeline_name}_score'] = score_tensor[idx].item()
            except Exception as e:
                logger.warning(f"Failed to process {pipeline_name} for idx {idx}: {str(e)}")
                row[f'{pipeline_name}'] = None

        if "accuracy" in scores:
            row['accuracy'] = scores["accuracy"][idx].item() if idx < len(scores["accuracy"]) else None

    def _populate_metrics(self, row: Dict, metrics: Dict) -> None:
        """Adds aggregate metric values to a row in place."""
        for pipeline_name, m in metrics.items():
            pipeline_name = pipeline_name.value if isinstance(pipeline_name, PipelineType) else pipeline_name
            if m.aucpr is not None:
                row[f'{pipeline_name}_aucpr'] = m.aucpr
            if m.aucroc is not None:
                row[f'{pipeline_name}_aucroc'] = m.aucroc
            if m.brier is not None:
                row[f'{pipeline_name}_brier'] = m.brier
            if m.mean_scores is not None:
                row[f'{pipeline_name}_mean'] = m.mean_scores

    def _write_results_to_excel(
        self,
        writer: pd.ExcelWriter,
        results: Union[EvaluationResults, GroupedResults],
        config: IntegratedPipelineConfig,
        dataset_entries: List[DatasetEntry],
        sheet_name: str,
        num_max_entries: Optional[int] = None
    ) -> None:
        """Writes all results to a single sheet."""
        # Prepare data for DataFrame
        rows = []

        if isinstance(results, GroupedResults):
            # Process each perturbation group
            for (pert_type, pert_intensity), group_results in results.grouped_results.items():
                # Filter entries for this perturbation group
                group_entries = [
                    entry for entry in dataset_entries
                    if entry.metadata.get('perturbation_type') == pert_type
                    and entry.metadata.get('perturbation_intensity') == pert_intensity
                ]

                # Process each entry in the group
                for idx, entry in enumerate(group_entries):
                    if num_max_entries and idx >= num_max_entries:
                        break
                    row = self._build_row(idx, entry, config, pert_type, pert_intensity)
                    self._populate_pipeline_scores(row, idx, group_results.scores)
                    self._populate_metrics(row, group_results.metrics)
                    rows.append(row)

        else:
            for idx, entry in enumerate(dataset_entries):
                if num_max_entries and idx >= num_max_entries:
                    break

                pert_type = entry.metadata.get('perturbation_type',
                    config.dataset_config.perturbation_type.value if hasattr(config.dataset_config, 'perturbation_type') else None)
                pert_intensity = entry.metadata.get('perturbation_intensity',
                    config.dataset_config.perturbation_intensity if hasattr(config.dataset_config, 'perturbation_intensity') else None)

                row = self._build_row(idx, entry, config, pert_type, pert_intensity)
                self._populate_pipeline_scores(row, idx, results.scores)
                self._populate_metrics(row, results.metrics)
                rows.append(row)

        # Create DataFrame and ensure we have data
        df = pd.DataFrame(rows)
        
        # Define column order
        ordered_columns = [
            'entry_idx',
            'question', 
            'answer',
            'accuracy',
            'full_decoded_text',
            'first_token_probability',
            'token_info_tuples',
            'effective_lengths',
            'padded_sequences',
            'semantic_entropy',
            'exp_id',
            'num_repeats',
            'dataset_name',
            'perturbation_type',
            'perturbation_intensity',
            'model',
            'generation_strategy',
            'prompt_strategy',
            'max_new_tokens'
        ]
        
        # Get remaining columns in their current order
        remaining_columns = [col for col in df.columns if col not in ordered_columns]
        
        # Combine ordered and remaining columns
        final_column_order = ordered_columns + remaining_columns
        
        # Reorder DataFrame columns
        df = df.reindex(columns=[col for col in final_column_order if col in df.columns])
        
        if not df.empty:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            
            # Auto-adjust column widths (existing code)
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = min(adjusted_width, 50)
            
    def _prepare_dataset(self, config: IntegratedPipelineConfig) -> DataLoader:
        """Prepares dataset for processing."""
        logger.info(f"Processing dataset of type {config.dataset_config.dataset_type}")
        processor = DatasetFactory.create_processor(
            dataset_type=config.dataset_config.dataset_type,
            merge_config=config.dataset_config.merge_config
        )
        
        return processor.process_dataset(config.dataset_config)
    
    def _prepare_dataloader(self, dataset: DatasetResult, batch_config: BatchConfig) -> DataLoader:
        """Prepares DataLoader for processing."""
        return DataLoader(
            queries=[entry.question for entry in dataset.entries],
            answers=np.array([entry.answer for entry in dataset.entries]),
            batch_config=batch_config
        )
        
    def _merge_batch_scores(
        self,
        accumulated: Dict[str, QuestionAggregateScores],
        new_scores: Dict[str, QuestionAggregateScores]
    ) -> None:
        """Merges new batch scores into accumulated scores."""
        logger.debug("Merging batch scores")
        for pipeline_name, scores in new_scores.items():
            if pipeline_name not in accumulated:
                accumulated[pipeline_name] = scores
                continue
            
            # Use ScoreAggregator to merge all levels of scores
            ScoreMerger.merge_question_scores(accumulated[pipeline_name], scores)

    def _extract_scores_and_metrics(
        self,
        accessor: ScoreAccessor,
        pipeline_types: List[PipelineType]
    ) -> EvaluationResults:
        """Extracts final scores and computes metrics."""
        logger.debug("Extracting scores and computing metrics")
        final_scores = {}
        metrics_dict = {}
        
        # Get token info first (they're pipeline-independent)
        token_level_fields = {
            'token_info': accessor.get_token_info_tuples,
            'full_text': accessor.get_full_decoded_text,
            'effective_lengths': accessor.get_effective_lengths,
            'padded_sequences': accessor.get_padded_sequences,
            'semantic_entropy': accessor.get_semantic_entropy_scores,
        }
        for pipeline_type in pipeline_types:
            config = TOKEN_ID_ACCESS_CONFIGS[pipeline_type]

            for field_name, getter in token_level_fields.items():
                if final_scores.get(field_name) is None:
                    value = getter(config.pipeline_name)
                    if value is not None:
                        final_scores[field_name] = value

            if all(final_scores.get(field_name) is not None for field_name in token_level_fields):
                break
                
        # Handle accuracy scores (rest of the existing code...)
        accuracy_configs = {pipeline_type: ACCURACY_ACCESS_CONFIGS[pipeline_type] for pipeline_type in pipeline_types}
        accuracy_values = None
        for config in accuracy_configs.values():
            if config.question_aggregate_type == QuestionAggregateTypes.ACCURACY:
                accuracy_values = accessor.get_score(
                    pipeline_name=config.pipeline_name,
                    token_score_type=None,
                    sequence_aggregate_type=None,
                    question_aggregate_type=QuestionAggregateTypes.ACCURACY
                )
                final_scores["accuracy"] = accuracy_values
                break
                
        # Process pipeline scores and metrics (existing code...)
        score_configs = {pipeline_type: SCORE_ACCESS_CONFIGS[pipeline_type] for pipeline_type in pipeline_types}
        for config in score_configs.values():
            if config.question_aggregate_type != QuestionAggregateTypes.ACCURACY:
                score = accessor.get_score(
                    pipeline_name=config.pipeline_name,
                    token_score_type=config.token_score_type,
                    sequence_aggregate_type=config.sequence_aggregate_type,
                    question_aggregate_type=config.question_aggregate_type
                )
                final_scores[config.pipeline_name] = score
                
                if config.pipeline_name in EVALUATION_PIPELINES_DICT:
                    pipeline_config = EVALUATION_PIPELINES_DICT[config.pipeline_name]
                    if hasattr(pipeline_config, 'dataset_metric_types'):
                        metrics_dict[config.pipeline_name] = self.metrics_calculator.compute_metrics(
                            score, accuracy_values
                        )
                        
        return EvaluationResults(scores=final_scores, metrics=metrics_dict)

    def _initialize_context(self, config: IntegratedPipelineConfig) -> EvaluationContext:
        """Initializes evaluation context with model and configurations."""
        logger.info(f"Loading model {config.model_identifier}")
        model, tokenizer = load_model_for_evaluation(
            model_identifier=config.model_identifier,
            device=config.device,
            max_memory=config.max_memory,
            apply_compile=config.apply_compile,
            model_save_path=config.model_save_path
        )
        
        return EvaluationContext(
            model=model,
            tokenizer=tokenizer,
            device=config.device,
            exp_id=config.exp_id,
            model_identifier=config.model_identifier,
            batch_config=config.batch_config,
            generation_config=config.generation_config,
            evaluation_config=config.evaluation_config,
            metrics_config=self.metrics_calculator.config
        )

    def _process_batches(
        self, 
        dataloader: DataLoader, 
        context: EvaluationContext
    ) -> Dict[str, QuestionAggregateScores]:
        """Processes all batches and aggregates scores."""
        logger.info("Processing batches")
        processor = BatchProcessor(context)
        accumulated_scores = {}
        
        for batch_idx, batch in enumerate(dataloader):
            batch_scores = processor.process_batch(batch)
            if not accumulated_scores:
                accumulated_scores = batch_scores
            else:
                self._merge_batch_scores(accumulated_scores, batch_scores)
            logger.debug(f"Completed batch {batch_idx + 1}")
            
        return accumulated_scores

    def _compute_final_results(
        self,
        evaluation_scores: Dict[str, QuestionAggregateScores],
        config: IntegratedPipelineConfig
    ) -> EvaluationResults:
        """Computes final evaluation results including metrics."""
        logger.info("Computing final results")
        accessor = ScoreAccessor(evaluation_scores)
        return self._extract_scores_and_metrics(accessor, config.pipeline_types)
