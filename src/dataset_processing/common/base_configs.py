from dataclasses import dataclass
from typing import Optional

from src.config import DATA_DIR
from src.dataset_processing.common.merge_config import MergeConfig
from src.dataset_processing.common.dataset_types import DatasetType
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.perturbations.enums import PerturbationType


@dataclass
class BaseDatasetConfig:
    """Base configuration shared by all datasets, for any source type (raw/processed/merged)."""
    dataset_type: DatasetType
    dataset_name: str
    source_type: DatasetSourceType
    base_dir: str = DATA_DIR
    num_entries: Optional[int] = None
    num_shots: Optional[int] = 0
    force_reprocess: bool = False
    merge_config: Optional[MergeConfig] = None
    random_seed: int = 42
    perturbation_type: PerturbationType = PerturbationType.NONE
    perturbation_intensity: int = 0
    num_perturbation_types: Optional[int] = None
