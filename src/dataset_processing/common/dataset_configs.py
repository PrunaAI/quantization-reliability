from dataclasses import dataclass
from typing import Optional

from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.dataset_types import DatasetType


@dataclass
class FKTCConfig(BaseDatasetConfig):
    """Configuration for FKTC datasets (any source type)."""
    num_relations: Optional[int] = 1

    def __post_init__(self):
        """Ensure dataset type is correct"""
        self.dataset_type = DatasetType.FKTC
        self.dataset_name = "fktc"


@dataclass
class TriviaQAConfig(BaseDatasetConfig):
    """Configuration for TriviaQA datasets (any source type)."""
    split: str = "validation"
    version: str = "rc.wikipedia"

    def __post_init__(self):
        """Ensure dataset type is correct"""
        self.dataset_type = DatasetType.TRIVIAQA
        self.dataset_name = "triviaqa"


@dataclass
class CommonSenseQAConfig(BaseDatasetConfig):
    """Configuration for CommonSenseQA datasets (any source type)."""
    split: str = "train"

    def __post_init__(self):
        """Ensure dataset type is correct"""
        self.dataset_type = DatasetType.COMMONSENSEQA
        self.dataset_name = "commonsenseqa"


@dataclass
class CoQAConfig(BaseDatasetConfig):
    """Configuration for CoQA datasets (any source type)."""
    split: str = "dev"
    questions_per_conversation: Optional[int] = None

    def __post_init__(self):
        """Ensure dataset type is correct"""
        self.dataset_type = DatasetType.COQA
        self.dataset_name = "coqa"


@dataclass
class MMLUConfig(BaseDatasetConfig):
    """Configuration for MMLU datasets (any source type)."""
    split: str = "test"
    subject: str = "abstract_algebra"

    def __post_init__(self):
        """Ensure dataset type is correct"""
        self.dataset_type = DatasetType.MMLU
        self.dataset_name = "mmlu"
