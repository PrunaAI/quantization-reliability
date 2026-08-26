from typing import Optional
from src.reliability_eval.common.score_types import QuestionAggregateTypes, SequenceAggregateTypes, TokenScoreTypes
from src.reliability_eval.common.access import ScoreAccessPath


class ScoreTypeMapper:
    """Maps enum score types to their corresponding dataclass attribute names."""
    
    _TOKEN_SCORE_MAP = {
        TokenScoreTypes.NLL: "negative_log_likelihood_scores",
        TokenScoreTypes.CROSS_ENTROPY: "cross_entropy_confidence_scores",
        TokenScoreTypes.ENTROPY: "entropy_scores",
        TokenScoreTypes.TOP_K: "top_k_concentration_scores",
        TokenScoreTypes.TOKEN_INFO_TUPLES: "token_info_tuples",
        TokenScoreTypes.FULL_DECODED_TEXT: "full_decoded_text",
        TokenScoreTypes.RELEVANT_DECODED_TEXT: "relevant_decoded_text",
        TokenScoreTypes.EFFECTIVE_LENGTHS: "effective_lengths",
        TokenScoreTypes.PADDED_SEQUENCES: "padded_sequences",
        TokenScoreTypes.SEMANTIC_ENTROPY: "semantic_entropy_scores",
    }
    
    _SEQUENCE_AGGREGATE_MAP = {
        SequenceAggregateTypes.MEAN: "mean_sequence_aggregate_scores",
        SequenceAggregateTypes.SUM: "total_sequence_aggregate_scores",
        SequenceAggregateTypes.MIN: "min_sequence_aggregate_scores",
        SequenceAggregateTypes.MAX: "max_sequence_aggregate_scores"
    }
    
    _QUESTION_AGGREGATE_MAP = {
        QuestionAggregateTypes.MEAN: "mean_question_aggregate_scores",
        QuestionAggregateTypes.SUM: "total_question_aggregate_scores",
        QuestionAggregateTypes.MIN: "min_question_aggregate_scores",
        QuestionAggregateTypes.MAX: "max_question_aggregate_scores",
        QuestionAggregateTypes.ACCURACY: "accuracy_aggregate_scores",
        QuestionAggregateTypes.MEAN_AMONG_CORRECT: "mean_among_correct_aggregate_scores"
    }

    @classmethod
    def get_access_path(
        cls,
        token_score_type: Optional[TokenScoreTypes],
        sequence_aggregate_type: Optional[SequenceAggregateTypes],
        question_aggregate_type: QuestionAggregateTypes
    ) -> ScoreAccessPath:
        """Returns the attribute path for accessing specific scores."""
        return ScoreAccessPath(
            question_aggregate=cls._QUESTION_AGGREGATE_MAP[question_aggregate_type],
            sequence_aggregate=cls._SEQUENCE_AGGREGATE_MAP.get(sequence_aggregate_type) if sequence_aggregate_type else None,
            token_score=cls._TOKEN_SCORE_MAP.get(token_score_type) if token_score_type else None
        )