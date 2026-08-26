from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import wandb
from src.loggers.wandb_utils import safe_wandb_log
from src.reliability_eval.common.evaluation import LLMEvaluationPipelineConfig
from src.reliability_eval.common.experiment import GenerationExperimentConfig
from src.reliability_eval.common.params import ExperimentConfigParam, GenerationConfigParam
from src.reliability_eval.common.constants import GENERATION_TERMINATOR_ID_DICT, model_name_to_model_family
from src.reliability_eval.common.batch import BatchData
from src.reliability_eval.common.scores import QuestionAggregateScores
from src.model_loading.common.tokenizer_utils import get_actual_tokenizer
from src.reliability_eval.evaluator.wrapper import ScoresCalculatorWrapper
from src.reliability_eval.generation.config import GenerationConfigHandler
from src.reliability_eval.prompting.strategies import PromptStrategyFactory
from src.reliability_eval.prompting.types import PromptStrategy
import logging

logger = logging.getLogger("reliability_eval")


class ModelGenerationEvaluator:
    """Main evaluator class for model generation reliability assessment."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,  # Still needed for terminator_ids lookup
        exp_id: str = None,
        device: str = "cuda"
    ):
        """Initialize evaluator with pre-loaded model and tokenizer."""
        logger.info(f"Initializing ModelGenerationEvaluator for model: {model_name}")
        
        # Store pre-loaded model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        
        # If device is 'auto', use cuda:0 for input tensors
        if device == 'auto':
            self.device = 'cuda:0'  # Use the first GPU for inputs
        else:
            self.device = device
        logger.info(f"Using pre-loaded model and tokenizer on device: {device}")
        
        # Set up configuration
        logger.debug("Setting up model configuration")
        self.terminator_ids = GENERATION_TERMINATOR_ID_DICT[model_name_to_model_family(model_name)]
        self.config_handler = GenerationConfigHandler(self.tokenizer)
        
        # Initialize score calculator
        logger.debug("Initializing scores calculator wrapper")
        self.scores_calculator_wrapper = ScoresCalculatorWrapper(
            model=self.model,
            tokenizer=self.tokenizer,
            exp_id=exp_id
        )
        logger.info("ModelGenerationEvaluator initialization complete")

    def evaluate_batch(
        self,
        queries: List[str],
        true_answers: np.array,
        generation_experiment_config: Optional[GenerationExperimentConfig] = None,
        evaluation_pipeline_dict: Optional[Dict[str, LLMEvaluationPipelineConfig]] = None
    ) -> Dict[str, QuestionAggregateScores]:
        """Evaluate model generations for a batch of queries."""
        logger.info(f"Starting batch evaluation for {len(queries)} queries")
        
        try:
            # Configure evaluation
            logger.debug("Configuring evaluation settings")
            self._configure_evaluation(generation_experiment_config, evaluation_pipeline_dict)
            # Process data
            logger.debug("Preparing batch data")
            processed_data = self._prepare_batch_data(queries, true_answers)
            
            # Generate outputs
            logger.debug("Generating model outputs")
            model_outputs = self._generate_model_outputs(processed_data)
            
            # Evaluate results
            logger.debug("Evaluating model outputs")
            evaluation_scores = self._evaluate_outputs(model_outputs, processed_data.repeated_answers, processed_data)
            
            logger.info("Batch evaluation completed successfully")
            return evaluation_scores
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {str(e)}")
            raise
        
    def _configure_evaluation(
        self,
        generation_experiment_config: Optional[GenerationExperimentConfig],
        evaluation_pipeline_dict: Optional[Dict[str, LLMEvaluationPipelineConfig]]
    ) -> None:
        """Set up generation and evaluation configurations."""
        logger.debug("Configuring evaluation parameters")
        try:
            self._set_generation_config(generation_experiment_config)
            self._set_evaluation_pipeline_dict(evaluation_pipeline_dict)
            logger.debug("Evaluation configuration complete")
        except Exception as e:
            logger.error(f"Evaluation configuration failed: {str(e)}")
            raise

    def _prepare_batch_data(
        self,
        queries: List[str],
        true_answers: np.array
    ) -> BatchData:
        """Prepare queries and answers for batch processing."""
        logger.debug(f"Preparing batch data for {len(queries)} queries")
        try:
            # Process queries and answers
            repeated_queries, repeated_answers = self._process_queries_and_answers(
                batched_queries=queries,
                true_answers=true_answers
            )
            logger.debug(f"Created {len(repeated_queries)} repeated queries")
            
            # Generate inputs
            inputs = self._generate_batch_inputs(repeated_queries).to(self.device)
            logger.debug("Generated batch inputs successfully")
            
            batch_data = BatchData(
                repeated_queries=repeated_queries,
                repeated_answers=repeated_answers,
                model_inputs=inputs
            )
            logger.debug("Batch data preparation complete")
            return batch_data
            
        except Exception as e:
            logger.error(f"Batch data preparation failed: {str(e)}")
            raise
        
    def _generate_model_outputs(self, batch_data: BatchData) -> torch.Tensor:
        """Generates model outputs for the prepared batch data."""
        logger.debug("Starting model output generation")
        try:
            outputs = self._generate_single_outputs(batch_data.model_inputs)
            logger.debug(f"Successfully generated outputs with shape {outputs.sequences.shape}")
            return outputs
        except Exception as e:
            logger.error(f"Failed to generate model outputs: {str(e)}")
            raise

    def _evaluate_outputs(
        self,
        outputs: torch.Tensor,
        repeated_answers: np.array,
        batch_data: BatchData  # Add batch_data parameter
    ) -> Dict[str, QuestionAggregateScores]:
        """Evaluates model outputs using configured metrics."""
        logger.debug(f"Starting output evaluation for {len(outputs)} outputs")
        evaluation_scores = {}
        
        try:
            # Extract unique queries (one per batch item)
            unique_queries = [query[0] for query in batch_data.repeated_queries]
            
            for config_name, config in self.evaluation_pipeline_dict.items():
                logger.debug(f"Evaluating with pipeline: {config_name}")
                scores = self.scores_calculator_wrapper.calculate_sequence_and_token_scores(
                    outputs=outputs,
                    repeated_answers=repeated_answers,
                    inference_config=self.inference_config,
                    evaluation_config=config,
                    queries=unique_queries  # Pass the original queries
                )
                evaluation_scores[config_name] = scores
                logger.debug(f"Completed evaluation for pipeline: {config_name}")
            
            logger.info(f"Successfully evaluated outputs using {len(self.evaluation_pipeline_dict)} pipelines")
            return evaluation_scores
        except Exception as e:
            logger.error(f"Failed to generate single outputs: {str(e)}")
            # Make sure to clean up even if there's an error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _generate_single_outputs(self, inputs) -> torch.Tensor:
        """Generate outputs for a batch of queries with retry mechanism."""
        MAX_RETRIES = 5
        BASE_DELAY = 1  # Base delay in seconds
        
        logger.debug("Generating single outputs")
        
        # Get actual tokenizer for VLM models
        actual_tokenizer = get_actual_tokenizer(self.tokenizer)
        
        for attempt in range(MAX_RETRIES):
            try:
                with torch.no_grad():
                    modified_inputs = inputs.copy() if attempt > 0 else inputs
                    modified_inference_config = self.inference_config
                    modified_inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in modified_inputs.items()}
                    
                    start_time = time.time()
                    outputs = self.model.generate(
                        **modified_inputs,
                        generation_config=GenerationConfig(**modified_inference_config),
                        tokenizer=self.tokenizer,
                        eos_token_id=actual_tokenizer.eos_token_id,
                        pad_token_id=actual_tokenizer.pad_token_id
                    )
                    
                    generation_time = time.time() - start_time
                    safe_wandb_log({"model_generation_time_seconds": generation_time})
                    
                    logger.debug(f"Successfully generated outputs with shape {outputs.sequences.shape}")
                    logger.debug(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    
                    return outputs
                    
            except RuntimeError as e:
                if "CUDA" in str(e) or "an illegal memory access was encountered" in str(e):
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed to generate single outputs after {MAX_RETRIES} attempts: {str(e)}")
                        raise
                    
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(f"CUDA memory error on attempt {attempt + 1}: {str(e)}. Retrying in {delay} seconds with reduced memory usage...")
                    time.sleep(delay)
                else:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed to generate single outputs after {MAX_RETRIES} attempts: {str(e)}")
                        raise
                    
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)

    def _apply_strategy_to_queries(self, queries: List[str]) -> List[str]:
        """Apply configured prompt strategy to queries."""
        strategy_type = self.inference_config.get(ExperimentConfigParam.PROMPT_STRATEGY.value)
        assert isinstance(strategy_type, PromptStrategy)
        dataset_name = self.inference_config.get(ExperimentConfigParam.DATASET_NAME.value)
        
        logger.debug(f"Applying prompt strategy '{strategy_type}' to {len(queries)} queries")
        
        try:
            processed_queries = [
                PromptStrategyFactory.apply_strategy(strategy_type, query, dataset_name)
                for query in queries
            ]
            logger.debug("Successfully applied prompt strategy")
            return processed_queries
        except Exception as e:
            logger.error(f"Failed to apply prompt strategy: {str(e)}")
            raise

    def _process_queries_and_answers(
        self,
        batched_queries: List[List[str]],
        true_answers: np.array
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process queries and answers for generation."""
        logger.debug(f"Processing {len(batched_queries)} queries and answers")
        try:
            batched_queries = self._apply_strategy_to_queries(batched_queries)
            queries_array = np.array(batched_queries)
            answers_array = np.array(true_answers)
            
            num_repeats = self.inference_config.get(ExperimentConfigParam.NUM_REPEATS.value, 1)
            num_return_sequences = self.inference_config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES.value, 1)
            
            logger.debug(f"Creating repeated arrays with {num_repeats} repeats and {num_return_sequences} return sequences")
            
            repeated_queries = np.tile(queries_array[:, np.newaxis], (1, num_repeats))
            repeated_answers = np.tile(answers_array[:, np.newaxis], (1, num_repeats * num_return_sequences))
            
            logger.debug(f"Created arrays with shapes: queries {repeated_queries.shape}, answers {repeated_answers.shape}")
            return repeated_queries, repeated_answers
        except Exception as e:
            logger.error(f"Failed to process queries and answers: {str(e)}")
            raise

    def _generate_batch_inputs(self, batched_queries: List[List[str]]) -> Dict:
        """Generate inputs for a batch of queries."""
        logger.debug(f"Generating batch inputs for {len(batched_queries)} queries")
        try:
            # Check if tokenizer is actually a processor (VLM)
            if hasattr(self.tokenizer, 'tokenizer'):
                # For VLM processor with text-only input
                inputs = self.tokenizer(
                    text=batched_queries.ravel().tolist(),
                    images=None,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
            else:
                # Standard tokenizer
                inputs = self.tokenizer(
                    batched_queries.ravel().tolist(),
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
            
            inputs['attention_mask'] = inputs['attention_mask'].bool()
            logger.debug(f"Generated inputs with shapes: {', '.join(f'{k}: {v.shape}' for k, v in inputs.items())}")
            return inputs
        except Exception as e:
            logger.error(f"Failed to generate batch inputs: {str(e)}")
            raise

    def _set_generation_config(
        self,
        generation_experiment_config: Optional[GenerationExperimentConfig]
    ) -> None:
        """Set up generation configuration using the config handler."""
        logger.debug("Setting up generation configuration")
        try:
            self.inference_config = self.config_handler.create_config(generation_experiment_config)
            logger.debug("Successfully created generation configuration")
        except Exception as e:
            logger.error(f"Failed to set generation configuration: {str(e)}")
            raise

    def _set_evaluation_pipeline_dict(self, evaluation_pipeline_dict: Optional[Dict[str, LLMEvaluationPipelineConfig]]):
        """Set the evaluation configuration for the evaluator."""
        logger.debug("Setting evaluation pipeline configuration")
        try:
            self.evaluation_pipeline_dict = evaluation_pipeline_dict if evaluation_pipeline_dict is not None else [LLMEvaluationPipelineConfig()]
            pipeline_count = len(self.evaluation_pipeline_dict) if evaluation_pipeline_dict else 1
            logger.debug(f"Set up {pipeline_count} evaluation pipeline(s)")
        except Exception as e:
            logger.error(f"Failed to set evaluation pipeline: {str(e)}")
            raise