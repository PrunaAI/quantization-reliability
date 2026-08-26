import os
import pandas as pd

from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.constants import DATA_FILES, FKTC_DATASET_FILES
from src.dataset_processing.common.source_types import DatasetSourceType


class FKTCFileHandler:
    """Handles file operations for FKTC datasets"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        
    def get_dataset_dir(self, source_type: DatasetSourceType) -> str:
        """Get appropriate directory based on source type"""
        return os.path.join(self.base_dir, "FKTC", source_type.value)

    def get_cache_path(self, cache_dir: str, config: BaseDatasetConfig) -> str:
        """Generate cache file path"""
        if config.source_type == DatasetSourceType.MERGED:
            return os.path.join(cache_dir, f"{config.dataset_name}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_rel-{str(config.num_relations)}_num-pert-{str(config.num_perturbation_types)}_merged.csv")
        elif config.source_type == DatasetSourceType.PROCESSED:
            return os.path.join(
                cache_dir,
                f"{config.dataset_name}_entries-{str(config.num_entries)}_shots-{str(config.num_shots)}_rel-{str(config.num_relations)}_pert-{config.perturbation_type.value.replace('_', '')}_intensity-{str(config.perturbation_intensity)}.csv"
            )
        elif config.source_type == DatasetSourceType.RAW:
            return os.path.join(cache_dir, f"{config.dataset_name}.csv")
        else:
            raise ValueError(f"Invalid source type: {config.source_type}")

    def read_all_datasets(self, num_relations: int) -> pd.DataFrame:
        """Read and merge all FKTC datasets"""
        dfs = []
        for dataset_name in FKTC_DATASET_FILES:
            df = self.read_csv_file(dataset_name, DatasetSourceType.RAW)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def read_csv_file(self, dataset_name: str, source_type: DatasetSourceType) -> pd.DataFrame:
        """Read and validate CSV file"""
        dir_path = self.get_dataset_dir(source_type)
        file_path = os.path.join(
            dir_path, 
            f"{dataset_name}.csv"
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        return pd.read_csv(file_path)
