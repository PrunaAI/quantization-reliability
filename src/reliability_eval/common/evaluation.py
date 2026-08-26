from dataclasses import dataclass, field
from typing import List

from src.reliability_eval.common.score_types import DatasetMetricTypes, QuestionAggregateTypes, SequenceAggregateTypes, SequenceScoreTypes, TokenScoreTypes


@dataclass
class LLMEvaluationPipelineConfig:
    """
    Dataclass to store evaluation configuration for language models.
    """
    token_score_types: List[TokenScoreTypes] = field(default_factory=lambda: [metric for metric in TokenScoreTypes])
    sequence_aggregate_types: List[SequenceAggregateTypes] = field(default_factory=lambda: [aggregate for aggregate in SequenceAggregateTypes])
    sequence_score_types: List[SequenceScoreTypes] = field(default_factory=lambda: [metric for metric in SequenceScoreTypes])
    question_aggregate_types: List[QuestionAggregateTypes] = field(default_factory=lambda: [aggregate for aggregate in QuestionAggregateTypes])
    dataset_metric_types: List[DatasetMetricTypes] = field(default_factory=lambda: [metric for metric in DatasetMetricTypes])
    