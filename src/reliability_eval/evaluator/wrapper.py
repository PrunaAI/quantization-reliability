import time
from typing import Dict, List

import torch
import numpy as np
import wandb
from src.loggers.wandb_utils import safe_wandb_log

from src.reliability_eval.calculators.aggregator import QuestionAggregator, SequenceAggregator
from src.reliability_eval.calculators.sequence import SequenceScoresCalculator
from src.reliability_eval.calculators.token import TokenScoresCalculator
from src.reliability_eval.common.evaluation import LLMEvaluationPipelineConfig
from src.reliability_eval.common.score_types import QuestionAggregateTypes, SequenceAggregateTypes, SequenceScoreTypes, TokenScoreTypes
from src.reliability_eval.common.scores import QuestionAggregateScores, SequenceAggregatedScores, SequenceScores, TokenScores


class ScoresCalculatorWrapper:
    """
    Wrapper class to calculate scores for a given model.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        exp_id: str = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.exp_id = exp_id
    
    def calculate_sequence_and_token_scores(
        self,
        outputs: torch.Tensor,
        repeated_answers: np.ndarray,
        inference_config: Dict,
        evaluation_config: LLMEvaluationPipelineConfig,
        queries: List[str] = None  # Add queries parameter
    ) -> List[Dict]:
        """
        Calculate sequence and sequence-level scores for each sequence.
        """
        self.set_up_scores_calculators(
            inference_config=inference_config
        )
        
        start_time = time.time()
        token_scores = self.calculate_token_scores(
            outputs=outputs,
            true_answers=repeated_answers,
            token_score_types=evaluation_config.token_score_types
        )
        token_scores_time = time.time() - start_time
        safe_wandb_log({"token_scores_calculation_time_seconds": token_scores_time})
        
        start_time = time.time()
        sequence_aggregate_scores = self.calculate_sequence_aggregate_scores(
            token_scores=token_scores,
            sequence_aggregate_types=evaluation_config.sequence_aggregate_types
        )
        sequence_agg_time = time.time() - start_time
        safe_wandb_log({"sequence_aggregation_time_seconds": sequence_agg_time})
        
        start_time = time.time()
        sequence_scores = self.calculate_sequence_scores(
            sequence_aggregate_scores=sequence_aggregate_scores,
            true_answers=repeated_answers,
            sequence_score_types=evaluation_config.sequence_score_types,
            queries=queries
        )
        sequence_scores_time = time.time() - start_time
        safe_wandb_log({"sequence_scores_calculation_time_seconds": sequence_scores_time})
        
        start_time = time.time()
        question_aggregate_scores = self.calculate_question_aggregate_scores(
            sequence_scores=sequence_scores,
            question_aggregate_types=evaluation_config.question_aggregate_types
        )
        question_agg_time = time.time() - start_time
        safe_wandb_log({"question_aggregation_time_seconds": question_agg_time})
        
        return question_aggregate_scores
    
    def calculate_token_scores(
        self,
        outputs: torch.Tensor,
        true_answers: np.array,
        token_score_types: List[TokenScoreTypes]
    ) -> TokenScores:
        """
        Calculate sequence-level scores for each sequence.
        """
        
        return self.token_scores_calculator.calculate_token_scores(
            outputs=outputs,
            true_answers=true_answers,
            token_score_types=token_score_types
        )
    
    def calculate_sequence_aggregate_scores(
        self,
        token_scores: TokenScores,
        sequence_aggregate_types: List[SequenceAggregateTypes]
    ) -> SequenceAggregatedScores:
        """
        Calculate sequence-level aggregate scores for each sequence.
        """
        
        return self.sequence_aggregator.calculate_sequence_aggregate_scores(
            token_scores=token_scores,
            sequence_aggregate_types=sequence_aggregate_types
        )
        
    def calculate_sequence_scores(
        self,
        sequence_aggregate_scores: SequenceAggregatedScores,
        true_answers: np.array,
        sequence_score_types: List[SequenceScoreTypes],
        queries: List[str] = None
    ) -> SequenceScores:
        """
        Calculate question-level scores for each sequence.
        """
        
        return self.sequence_scores_calculator.calculate_sequence_scores(
            sequence_aggregate_scores=sequence_aggregate_scores,
            sequence_score_types=sequence_score_types,
            true_answers=true_answers,
            queries=queries
        )
        
    def calculate_question_aggregate_scores(
        self,
        sequence_scores: SequenceScores,
        question_aggregate_types: List[QuestionAggregateTypes]
    ) -> QuestionAggregateScores:
        """
        Calculate question-level aggregate scores for each sequence.
        """
        
        return self.question_aggregator.calculate_question_aggregate_scores(
            sequence_scores=sequence_scores,
            question_aggregate_types=question_aggregate_types
        )
        
    def set_up_scores_calculators(
        self,
        inference_config: Dict
    ):
        """
        Set up the scores calculators for the ResponseGenerator.
        """
        self.token_scores_calculator = TokenScoresCalculator(
            model=self.model,
            tokenizer=self.tokenizer,
            inference_config=inference_config,
            exp_id=self.exp_id
        )
        
        self.sequence_aggregator = SequenceAggregator(
            model=self.model,
            tokenizer=self.tokenizer,
            inference_config=inference_config,
        )
        
        self.sequence_scores_calculator = SequenceScoresCalculator(
            model=self.model,
            tokenizer=self.tokenizer,
            inference_config=inference_config,
        )
        
        self.question_aggregator = QuestionAggregator(
            model=self.model,
            tokenizer=self.tokenizer,
            inference_config=inference_config,
        )
