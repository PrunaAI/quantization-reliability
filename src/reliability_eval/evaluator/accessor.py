
from typing import Any, Dict, List, Optional, Tuple

import torch
from src.reliability_eval.common.score_types import QuestionAggregateTypes, SequenceAggregateTypes, TokenScoreTypes
from src.reliability_eval.common.access import ScoreAccessPath
from src.reliability_eval.common.scores import QuestionAggregateScores
from src.reliability_eval.evaluator.mapper import ScoreTypeMapper


class ScoreAccessor:
    """Provides easy access to nested evaluation scores."""

    def __init__(self, evaluation_scores: Dict[str, QuestionAggregateScores]):
        """Initializes accessor with evaluation scores dictionary."""
        self.evaluation_scores = evaluation_scores
        self.mapper = ScoreTypeMapper()
    
    def _get_from_mean_seq(self, pipeline_name: str, attr: str) -> Optional[Any]:
        """Navigate to mean_sequence_aggregate_scores and return the given attribute."""
        if pipeline_name not in self.evaluation_scores:
            return None
        scores = self.evaluation_scores[pipeline_name]
        if not hasattr(scores, 'mean_question_aggregate_scores'):
            return None
        question_scores = scores.mean_question_aggregate_scores
        if not hasattr(question_scores, 'mean_sequence_aggregate_scores'):
            return None
        sequence_scores = question_scores.mean_sequence_aggregate_scores
        return getattr(sequence_scores, attr, None)

    def get_token_info_tuples(self, pipeline_name: str) -> Optional[List[List[List[Tuple[int, str, float]]]]]:
        """Get token info tuples from any pipeline that has them."""
        return self._get_from_mean_seq(pipeline_name, 'token_info_tuples')

    def get_full_decoded_text(self, pipeline_name: str) -> Optional[List[List[str]]]:
        """Get full decoded text from any pipeline that has it."""
        return self._get_from_mean_seq(pipeline_name, 'full_decoded_text')

    def get_effective_lengths(self, pipeline_name: str) -> Optional[List[List[int]]]:
        """Get effective lengths from any pipeline that has them."""
        return self._get_from_mean_seq(pipeline_name, 'effective_lengths')

    def get_padded_sequences(self, pipeline_name: str) -> Optional[List[List[List[int]]]]:
        """Get padded sequences from any pipeline that has them."""
        return self._get_from_mean_seq(pipeline_name, 'padded_sequences')

    def get_semantic_entropy_scores(self, pipeline_name: str) -> Optional[torch.Tensor]:
        """Get semantic entropy scores from any pipeline that has them."""
        return self._get_from_mean_seq(pipeline_name, 'semantic_entropy_scores')
    
    def get_score(
        self,
        pipeline_name: str,
        token_score_type: TokenScoreTypes,
        sequence_aggregate_type: SequenceAggregateTypes,
        question_aggregate_type: QuestionAggregateTypes
    ) -> Optional[torch.Tensor]:
        """Returns tensor score for specified score types and aggregation methods."""
        access_path = self.mapper.get_access_path(
            token_score_type,
            sequence_aggregate_type,
            question_aggregate_type
        )
        return self._traverse_score_path(pipeline_name, access_path)

    def _safe_getattr(self, obj: Any, attr: str) -> Optional[Any]:
        """Safely gets attribute value, returns None if attribute doesn't exist or is None."""
        if obj is None:
            return None
        return getattr(obj, attr, None)

    def _traverse_score_path(
        self,
        pipeline_name: str,
        access_path: ScoreAccessPath
    ) -> Optional[torch.Tensor]:
        """Traverses nested score objects to retrieve final tensor score."""
        score_obj = self.evaluation_scores.get(pipeline_name)
        if score_obj is None:
            return None
            
        if access_path.question_aggregate == "accuracy_aggregate_scores":
            return self._safe_getattr(score_obj, access_path.question_aggregate)
            
        score_obj = self._safe_getattr(score_obj, access_path.question_aggregate)
        if score_obj is None:
            return None
            
        if access_path.sequence_aggregate:
            score_obj = self._safe_getattr(score_obj, access_path.sequence_aggregate)
            if score_obj is None:
                return None
            
        if access_path.token_score:
            score_obj = self._safe_getattr(score_obj, access_path.token_score)
            
        return score_obj