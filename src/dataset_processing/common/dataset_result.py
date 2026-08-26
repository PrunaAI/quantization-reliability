from typing import List, Optional

from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig


class DatasetResult:
    """Container for dataset loading results"""
    def __init__(
        self,
        entries: List[DatasetEntry],
        perturbation_config: Optional[List[PerturbationConfig]] = None,
        config: Optional[BaseDatasetConfig] = None
    ):
        self.entries = entries
        self.perturbation_config = perturbation_config
        self.config = config
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        return self.entries[index]