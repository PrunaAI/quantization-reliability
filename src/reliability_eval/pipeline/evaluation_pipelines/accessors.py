from src.reliability_eval.common.score_types import QuestionAggregateTypes, SequenceAggregateTypes, TokenScoreTypes
from src.reliability_eval.pipeline.evaluation_pipelines.types import PipelineType
from src.reliability_eval.pipeline.config import ScoreAccessConfig


_TOKEN_SCORE_TYPES = {
    PipelineType.NLL: TokenScoreTypes.NLL,
    PipelineType.CONFIDENCE: TokenScoreTypes.CROSS_ENTROPY,
    PipelineType.ENTROPY: TokenScoreTypes.ENTROPY,
    PipelineType.TOPK: TokenScoreTypes.TOP_K,
    PipelineType.SEMANTIC_ENTROPY: TokenScoreTypes.SEMANTIC_ENTROPY,
}

SCORE_ACCESS_CONFIGS = {
    pipeline_type: ScoreAccessConfig(
        pipeline_name=pipeline_type,
        token_score_type=token_score_type,
        sequence_aggregate_type=SequenceAggregateTypes.MEAN,
        question_aggregate_type=QuestionAggregateTypes.MEAN
    )
    for pipeline_type, token_score_type in _TOKEN_SCORE_TYPES.items()
}

ACCURACY_ACCESS_CONFIGS = {
    pipeline_type: ScoreAccessConfig(
        pipeline_name=pipeline_type,
        token_score_type=None,
        sequence_aggregate_type=None,
        question_aggregate_type=QuestionAggregateTypes.ACCURACY
    )
    for pipeline_type in _TOKEN_SCORE_TYPES
}

TOKEN_ID_ACCESS_CONFIGS = {
    pipeline_type: ScoreAccessConfig(
        pipeline_name=pipeline_type,
        token_score_type=token_score_type,
        sequence_aggregate_type=None,
        question_aggregate_type=None
    )
    for pipeline_type, token_score_type in _TOKEN_SCORE_TYPES.items()
}
