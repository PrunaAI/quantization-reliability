import os
import datasets
import pandas as pd
from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.source_types import DatasetSourceType


class MMLUFileHandler:
    """Handles file operations for MMLU datasets."""
    
    def __init__(self, base_dir: str):
        """Initialize file handler with base directory."""
        self.base_dir = base_dir

    def get_dataset_dir(self, source_type: DatasetSourceType) -> str:
        """Get appropriate directory based on source type."""
        return os.path.join(self.base_dir, "MMLU", source_type.value)

    def get_cache_path(self, cache_dir: str, config: BaseDatasetConfig) -> str:
        """Generate cache file path."""
        if config.source_type == DatasetSourceType.MERGED:
            return os.path.join(
                cache_dir,
                f"mmlu_subject-{config.subject}_split-{config.split}_entries-{str(config.num_entries)}_num-pert-{str(config.num_perturbation_types)}_merged.csv"
            )
        elif config.source_type == DatasetSourceType.PROCESSED:
            return os.path.join(
                cache_dir,
                f"mmlu_subject-{config.subject}_split-{config.split}_entries-{str(config.num_entries)}_pert-{config.perturbation_type.value.replace('_', '')}_intensity-{str(config.perturbation_intensity)}.csv"
            )
        else:
            raise ValueError(f"Invalid source type: {config.source_type}")

    def load_dataset(self, split: str, subject: str) -> pd.DataFrame:
        """Load dataset from raw files, downloading if necessary."""
        file_path = os.path.join(self.base_dir, "MMLU", "raw", subject, f"{split}.csv")
        
        if not os.path.exists(file_path):
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Download from Hugging Face
            dataset = datasets.load_dataset("cais/mmlu", subject, split=split)
            
            # First convert choices to separate columns
            data = []
            for item in dataset:
                row = {
                    'question': item['question'],
                    'A': item['choices'][0],
                    'B': item['choices'][1],
                    'C': item['choices'][2],
                    'D': item['choices'][3],
                    'answer': item['answer']
                }
                data.append(row)
                
            # Create DataFrame from list of dictionaries
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            return df
            
        return pd.read_csv(file_path)