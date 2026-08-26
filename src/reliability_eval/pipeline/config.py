from dataclasses import dataclass
from typing import Dict, List, Optional

from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.model_loading.common.identifier import ModelIdentifier
from src.reliability_eval.common.evaluation import LLMEvaluationPipelineConfig
from src.reliability_eval.common.experiment import GenerationExperimentConfig
from src.reliability_eval.common.score_types import QuestionAggregateTypes, SequenceAggregateTypes, TokenScoreTypes
from src.reliability_eval.pipeline.evaluation_pipelines.types import PipelineType


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    shuffle: bool = False
    drop_last: bool = False

@dataclass
class ScoreAccessConfig:
    """Configuration for accessing specific score types."""
    pipeline_name: str
    token_score_type: TokenScoreTypes
    sequence_aggregate_type: SequenceAggregateTypes
    question_aggregate_type: QuestionAggregateTypes

@dataclass
class IntegratedPipelineConfig:
    """Configuration for the complete evaluation pipeline."""
    model_identifier: ModelIdentifier
    dataset_config: BaseDatasetConfig
    generation_config: GenerationExperimentConfig
    evaluation_config: Dict[str, LLMEvaluationPipelineConfig]
    num_excel_entries: int
    pipeline_types: List[PipelineType]
    exp_id: str
    batch_config: BatchConfig = BatchConfig()
    device: str = "cuda"
    max_memory: Optional[Dict[str, str]] = None
    apply_compile: bool = True
    random_seed: int = 42
    model_save_path: Optional[str] = None
