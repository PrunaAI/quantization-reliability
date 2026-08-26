from typing import Any

import torch
import logging

from src.reliability_eval.common.score_types import QuestionAggregateTypes, SequenceAggregateTypes, TokenScoreTypes
from src.reliability_eval.common.scores import QuestionAggregateScores, SequenceScores, TokenScores
from src.reliability_eval.evaluator.mapper import ScoreTypeMapper


logger = logging.getLogger("reliability_eval")

class ScoreMerger:
    """Handles merging of scores across batches."""
    
    @staticmethod
    def merge_token_scores(accumulated: TokenScores, new_scores: TokenScores) -> None:
        """Merges token-level scores."""
        logger.debug("Merging token scores")
        for token_type in TokenScoreTypes:
            attr_name = ScoreTypeMapper._TOKEN_SCORE_MAP[token_type]
            ScoreMerger._merge_attribute(accumulated, new_scores, attr_name)
    
    @staticmethod
    def merge_sequence_scores(accumulated: SequenceScores, new_scores: SequenceScores) -> None:
        """Merges sequence-level scores."""
        logger.debug("Merging sequence scores")
        # Merge sequence aggregated scores
        for seq_type in SequenceAggregateTypes:
            attr_name = ScoreTypeMapper._SEQUENCE_AGGREGATE_MAP[seq_type]
            acc_scores = getattr(accumulated, attr_name, None)
            new_score = getattr(new_scores, attr_name, None)
            
            if acc_scores is not None and new_score is not None:
                ScoreMerger.merge_token_scores(acc_scores, new_score)
        
        # Merge correctness boolean values
        if accumulated.is_correct is not None and new_scores.is_correct is not None:
            accumulated.is_correct = torch.cat([accumulated.is_correct, new_scores.is_correct], dim=0)
    
    @staticmethod
    def merge_question_scores(accumulated: QuestionAggregateScores, new_scores: QuestionAggregateScores) -> None:
        """Merges question-level scores."""
        logger.debug("Merging question scores")
        for question_type in QuestionAggregateTypes:
            attr_name = ScoreTypeMapper._QUESTION_AGGREGATE_MAP[question_type]
            acc_scores = getattr(accumulated, attr_name, None)
            new_score = getattr(new_scores, attr_name, None)
            
            if acc_scores is not None and new_score is not None:
                if question_type == QuestionAggregateTypes.ACCURACY:
                    # Special handling for accuracy scores which are direct tensors
                    setattr(accumulated, attr_name,
                        torch.cat([acc_scores, new_score], dim=0))
                else:
                    # Handle sequence scores which contain token scores
                    ScoreMerger.merge_sequence_scores(acc_scores, new_score)
    
    @staticmethod
    def _merge_attribute(accumulated: Any, new_scores: Any, attr_name: str) -> None:
        """Merges a single attribute between two score objects."""
        acc_value = getattr(accumulated, attr_name, None)
        new_value = getattr(new_scores, attr_name, None)
        
        if acc_value is not None and new_value is not None:
            if isinstance(acc_value, torch.Tensor):
                setattr(accumulated, attr_name, torch.cat([acc_value, new_value], dim=0))
            elif isinstance(acc_value, list):
                acc_value.extend(new_value)
