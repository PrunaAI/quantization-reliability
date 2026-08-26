from typing import Dict, List, Optional, Union

import torch
from src.reliability_eval.pipeline.results import EvaluationResults, GroupedResults, MetricResults, ProcessedMetrics, ProcessedResult


class ResultsProcessor:
    """Handles post-processing of evaluation results for database storage."""
    
    @staticmethod
    def _process_metrics(metrics: Dict[str, MetricResults]) -> Dict[str, ProcessedMetrics]:
        """Processes metrics for a single EvaluationResults instance."""
        return {
            pipeline_name: ProcessedMetrics(
                aucpr=metric.aucpr,
                aucroc=metric.aucroc,
                brier=metric.brier,
                mean_scores=metric.mean_scores
            )
            for pipeline_name, metric in metrics.items()
        }
    
    @staticmethod
    def _process_scores(scores: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
        """Processes scores for a single EvaluationResults instance."""
        return {
            pipeline_name: tensor.cpu().tolist() if isinstance(tensor, torch.Tensor) else tensor
            for pipeline_name, tensor in scores.items()
        }
    
    @staticmethod
    def _process_evaluation_results(
        results: EvaluationResults,
        perturbation_type: Optional[str] = None,
        perturbation_intensity: Optional[int] = None
    ) -> ProcessedResult:
        """Processes a single EvaluationResults instance."""
        return ProcessedResult(
            perturbation_type=perturbation_type,
            perturbation_intensity=perturbation_intensity,
            pipeline_scores=ResultsProcessor._process_scores(results.scores),
            pipeline_metrics=ResultsProcessor._process_metrics(results.metrics)
        )
    
    @classmethod
    def process_results(
        cls,
        results: Union[EvaluationResults, GroupedResults]
    ) -> List[ProcessedResult]:
        """Processes evaluation results into a database-friendly format."""
        if isinstance(results, GroupedResults):
            processed_results = [
                cls._process_evaluation_results(
                    results.overall_results,
                    perturbation_type="overall",
                    perturbation_intensity=None
                )
            ]
            
            processed_results.extend([
                cls._process_evaluation_results(
                    group_results,
                    perturbation_type=pert_type,
                    perturbation_intensity=pert_intensity
                )
                for (pert_type, pert_intensity), group_results in results.grouped_results.items()
            ])
            
            return processed_results
        else:
            return [cls._process_evaluation_results(results)]

def post_process_results(results: Union[EvaluationResults, GroupedResults]) -> Dict[str, List]:
    """Creates a flattened summary dictionary from processed results."""
    processed_results = ResultsProcessor.process_results(results)
    
    summary = {
        "perturbation_types": [],
        "perturbation_intensities": []
    }
    
    results_to_process = processed_results[1:] if isinstance(results, GroupedResults) else processed_results
    is_single_result = len(processed_results) == 1
    
    for result in results_to_process:
        # Handle default values for single results
        if is_single_result:
            summary["perturbation_types"].append(result.perturbation_type or "none")
            summary["perturbation_intensities"].append(result.perturbation_intensity or 0)
        else:
            if result.perturbation_type:
                summary["perturbation_types"].append(result.perturbation_type)
            if result.perturbation_intensity is not None:
                summary["perturbation_intensities"].append(result.perturbation_intensity)
            
        # Flatten scores to single-level lists
        for pipeline_name, scores in result.pipeline_scores.items():
            pipeline_key = pipeline_name.value if hasattr(pipeline_name, 'value') else pipeline_name
            score_key = f"scores.{pipeline_key}"
            if score_key not in summary:
                summary[score_key] = []
            # Ensure scores is a flat list
            flat_scores = scores if isinstance(scores, list) else [scores]
            summary[score_key].extend(flat_scores)
            
        for pipeline_name, metrics in result.pipeline_metrics.items():
            pipeline_key = pipeline_name.value if hasattr(pipeline_name, 'value') else pipeline_name
            for metric_name in ['aucpr', 'aucroc', 'brier', 'mean_scores']:
                metric_key = f"metrics.{pipeline_key}.{metric_name}"
                if metric_key not in summary:
                    summary[metric_key] = []
                metric_value = getattr(metrics, metric_name)
                summary[metric_key].append(metric_value)
    
    # Remove these lines to include all token IDs and decoded text
    summary["scores.token_info"] = summary["scores.token_info"][:50]
    summary["scores.full_text"] = summary["scores.full_text"][:50]
    
    return summary
