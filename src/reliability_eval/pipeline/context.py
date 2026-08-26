from dataclasses import dataclass
from typing import Any, Dict, List

from src.model_loading.common.identifier import ModelIdentifier
from src.reliability_eval.common.evaluation import LLMEvaluationPipelineConfig
from src.reliability_eval.common.experiment import GenerationExperimentConfig
from src.reliability_eval.common.score_types import DatasetMetricTypes
from src.reliability_eval.pipeline.config import BatchConfig


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""
    metric_types: List[DatasetMetricTypes]
    use_exponential: bool = True
    round_accuracy: bool = True
    

@dataclass
class EvaluationContext:
    """Context for evaluation process containing config and resources."""
    model: Any
    tokenizer: Any
    device: str
    exp_id: str
    model_identifier: ModelIdentifier  # Added this field
    batch_config: BatchConfig
    generation_config: GenerationExperimentConfig
    evaluation_config: Dict[str, LLMEvaluationPipelineConfig]
    metrics_config: MetricsConfig
