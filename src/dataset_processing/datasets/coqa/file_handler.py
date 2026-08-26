import json
import os
from typing import Dict
from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.source_types import DatasetSourceType


class CoQAFileHandler:
    """Handles file operations for CoQA datasets."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        
    def get_dataset_dir(self, source_type: DatasetSourceType) -> str:
        """Get appropriate directory based on source type."""
        return os.path.join(self.base_dir, "CoQA", source_type.value)
    
    def get_cache_path(self, cache_dir: str, config: BaseDatasetConfig) -> str:
        """Generate cache file path."""
        if config.source_type == DatasetSourceType.PROCESSED:
            return os.path.join(
                cache_dir,
                f"coqa_split-{config.split}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_pert-{config.perturbation_type.value.replace('_', '')}_intensity-{str(config.perturbation_intensity)}.csv"
            )
        elif config.source_type == DatasetSourceType.MERGED:
            return os.path.join(
                cache_dir,
                f"coqa_split-{config.split}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_num-pert-{str(config.num_perturbation_types)}_merged.csv"
            )
        elif config.source_type == DatasetSourceType.RAW:
            return os.path.join(
                cache_dir,
                f"coqa_split-{config.split}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_raw.csv"
            )
        else: # Invalid source type
            raise ValueError(f"Invalid source type: {config.source_type}")
    
    def read_json_file(self, source_type: DatasetSourceType, split: str) -> Dict:
        """Read and validate JSON file."""
        dir_path = self.get_dataset_dir(source_type)
        file_path = os.path.join(dir_path, f"coqa-{split}-v1.0.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not data or 'data' not in data:
            raise ValueError(f"Invalid data format in file: {file_path}")
            
        return data
