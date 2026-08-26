from typing import Dict
import logging
from src.reliability_eval.common.scores import QuestionAggregateScores
from src.reliability_eval.evaluator.model_evaluator import ModelGenerationEvaluator
from src.reliability_eval.pipeline.context import EvaluationContext
from src.reliability_eval.pipeline.processor.batch import Batch
from src.reliability_eval.pipeline.processor.merger import ScoreMerger


logger = logging.getLogger("reliability_eval")

class BatchProcessor:
    """Handles processing of individual batches."""
    
    def __init__(self, context: EvaluationContext):
        """Initialize with evaluation context."""
        self.context = context
        self.evaluator = ModelGenerationEvaluator(
            model=context.model,
            tokenizer=context.tokenizer,
            model_name=str(context.model_identifier),
            device=context.device,
            exp_id=context.exp_id
        )
        self.score_aggregator = ScoreMerger()

    def process_batch(self, batch: Batch) -> Dict[str, QuestionAggregateScores]:
        """Process a single batch and return scores."""
        logger.debug(f"Processing batch with {len(batch.queries)} items")
        return self.evaluator.evaluate_batch(
            queries=batch.queries,
            true_answers=batch.answers,
            generation_experiment_config=self.context.generation_config,
            evaluation_pipeline_dict=self.context.evaluation_config
        )
