import os

import datasets

from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.source_types import DatasetSourceType


class CommonSenseQAFileHandler:
    """Handles file operations for CommonSenseQA datasets."""
    
    def __init__(self, base_dir: str):
        """Initialize file handler with base directory."""
        self.base_dir = base_dir
        
    def get_dataset_dir(self, source_type: DatasetSourceType) -> str:
        """Get appropriate directory based on source type."""
        return os.path.join(self.base_dir, "CommonSenseQA", source_type.value)
    
    def get_cache_path(self, cache_dir: str, config: BaseDatasetConfig) -> str:
        """Generate cache file path."""
        if config.source_type == DatasetSourceType.MERGED:
            return os.path.join(
                cache_dir,
                f"commonsenseqa_split-{config.split}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_num-pert-{str(config.num_perturbation_types)}_merged.csv"
            )
        elif config.source_type == DatasetSourceType.PROCESSED:
            return os.path.join(
                cache_dir,
                f"commonsenseqa_split-{config.split}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_pert-{config.perturbation_type.value.replace('_', '')}_intensity-{str(config.perturbation_intensity)}.csv"
            )
        elif config.source_type == DatasetSourceType.RAW:  # Add this case
            return os.path.join(
                cache_dir,
                f"commonsenseqa_split-{config.split}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_raw.csv"
            )
        else:  # Invalid source type
            raise ValueError(f"Invalid source type: {config.source_type}")
    
    def load_huggingface_dataset(self, split: str) -> datasets.Dataset:
        """Load dataset from HuggingFace."""
        return datasets.load_dataset("tau/commonsense_qa", split=split)
