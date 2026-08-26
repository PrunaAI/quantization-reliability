from src.reliability_eval.common.score_types import (
    TokenScoreTypes, SequenceAggregateTypes,
    SequenceScoreTypes, QuestionAggregateTypes,
    DatasetMetricTypes
)
from src.reliability_eval.common.evaluation import LLMEvaluationPipelineConfig

CONFIDENCE_EVALUATION_PIPELINE = LLMEvaluationPipelineConfig(
    token_score_types=[
        TokenScoreTypes.CROSS_ENTROPY,
        TokenScoreTypes.TOKEN_INFO_TUPLES,
        TokenScoreTypes.RELEVANT_DECODED_TEXT,
        TokenScoreTypes.FULL_DECODED_TEXT,
        TokenScoreTypes.EFFECTIVE_LENGTHS,
        TokenScoreTypes.PADDED_SEQUENCES
    ],
    sequence_aggregate_types=[
        SequenceAggregateTypes.MEAN
    ],
    sequence_score_types=[
        SequenceScoreTypes.ID,
        SequenceScoreTypes.IS_CORRECT
    ],
    question_aggregate_types=[
        QuestionAggregateTypes.MEAN,
        QuestionAggregateTypes.ACCURACY
    ],
    dataset_metric_types=[
        DatasetMetricTypes.ACCURACY,
        DatasetMetricTypes.AUCPR,
        DatasetMetricTypes.AUCROC,
        DatasetMetricTypes.BRIER,
        DatasetMetricTypes.MEAN_SCORES
    ]
)

NLL_EVALUATION_PIPELINE = LLMEvaluationPipelineConfig(
    token_score_types=[
        TokenScoreTypes.NLL,
        TokenScoreTypes.TOKEN_INFO_TUPLES,
        TokenScoreTypes.RELEVANT_DECODED_TEXT,
        TokenScoreTypes.FULL_DECODED_TEXT,
        TokenScoreTypes.EFFECTIVE_LENGTHS,
        TokenScoreTypes.PADDED_SEQUENCES
    ],
    sequence_aggregate_types=[
        SequenceAggregateTypes.MEAN
    ],
    sequence_score_types=[
        SequenceScoreTypes.ID,
        SequenceScoreTypes.IS_CORRECT
    ],
    question_aggregate_types=[
        QuestionAggregateTypes.MEAN,
        QuestionAggregateTypes.ACCURACY
    ],
    dataset_metric_types=[
        DatasetMetricTypes.ACCURACY,
        DatasetMetricTypes.AUCPR,
        DatasetMetricTypes.AUCROC,
        DatasetMetricTypes.BRIER,
        DatasetMetricTypes.MEAN_SCORES
    ]
)

ENTROPY_EVALUATION_PIPELINE = LLMEvaluationPipelineConfig(
    token_score_types=[
        TokenScoreTypes.ENTROPY,
        TokenScoreTypes.TOKEN_INFO_TUPLES,
        TokenScoreTypes.RELEVANT_DECODED_TEXT,
        TokenScoreTypes.FULL_DECODED_TEXT,
        TokenScoreTypes.EFFECTIVE_LENGTHS,
        TokenScoreTypes.PADDED_SEQUENCES
    ],
    sequence_aggregate_types=[
        SequenceAggregateTypes.MEAN
    ],
    sequence_score_types=[
        SequenceScoreTypes.ID,
        SequenceScoreTypes.IS_CORRECT
    ],
    question_aggregate_types=[
        QuestionAggregateTypes.MEAN,
        QuestionAggregateTypes.ACCURACY
    ],
    dataset_metric_types=[
        DatasetMetricTypes.ACCURACY,
        DatasetMetricTypes.AUCPR,
        DatasetMetricTypes.AUCROC,
        DatasetMetricTypes.MEAN_SCORES
    ]
)


TOP_K_EVALUATION_PIPELINE = LLMEvaluationPipelineConfig(
    token_score_types=[
        TokenScoreTypes.TOP_K,
        TokenScoreTypes.TOKEN_INFO_TUPLES,
        TokenScoreTypes.RELEVANT_DECODED_TEXT,
        TokenScoreTypes.FULL_DECODED_TEXT,
        TokenScoreTypes.EFFECTIVE_LENGTHS,
        TokenScoreTypes.PADDED_SEQUENCES
    ],
    sequence_aggregate_types=[
        SequenceAggregateTypes.MEAN
    ],
    sequence_score_types=[
        SequenceScoreTypes.ID,
        SequenceScoreTypes.IS_CORRECT
    ],
    question_aggregate_types=[
        QuestionAggregateTypes.MEAN,
        QuestionAggregateTypes.ACCURACY
    ],
    dataset_metric_types=[
        DatasetMetricTypes.ACCURACY,
        DatasetMetricTypes.AUCPR,
        DatasetMetricTypes.AUCROC,
        DatasetMetricTypes.MEAN_SCORES
    ]
)


SEMANTIC_ENTROPY_EVALUATION_PIPELINE = LLMEvaluationPipelineConfig(
    token_score_types=[
        TokenScoreTypes.SEMANTIC_ENTROPY,
        TokenScoreTypes.TOKEN_INFO_TUPLES,
        TokenScoreTypes.RELEVANT_DECODED_TEXT,
        TokenScoreTypes.FULL_DECODED_TEXT,
        TokenScoreTypes.EFFECTIVE_LENGTHS,
        TokenScoreTypes.PADDED_SEQUENCES
    ],
    sequence_aggregate_types=[
        SequenceAggregateTypes.MEAN
    ],
    sequence_score_types=[
        SequenceScoreTypes.ID,
        SequenceScoreTypes.IS_CORRECT
    ],
    question_aggregate_types=[
        QuestionAggregateTypes.MEAN,
        QuestionAggregateTypes.ACCURACY
    ],
    dataset_metric_types=[
        DatasetMetricTypes.ACCURACY,
        DatasetMetricTypes.AUCPR,
        DatasetMetricTypes.AUCROC,
        DatasetMetricTypes.MEAN_SCORES
    ]
)