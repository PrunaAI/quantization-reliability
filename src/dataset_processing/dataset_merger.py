import os
import pandas as pd
import logging

from dataclasses import replace
from typing import Optional, Union
from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.merge_config import MergeConfig
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.common.dataset_result import DatasetResult
from src.dataset_processing.common.base_processor import DatasetProcessor
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.enums import PerturbationType

logger = logging.getLogger("reliability_eval")


class DatasetMerger:
    """Handles merging of processed datasets with different perturbations"""

    def __init__(self, processor: Union[DatasetProcessor], merge_config: Optional[MergeConfig] = None):
        self.processor = processor
        self.merge_config = merge_config or MergeConfig()
    
    def ensure_processed_datasets(self, config: BaseDatasetConfig) -> None:
        """Ensure all required processed datasets exist"""
        logger.info(f"Ensuring processed datasets exist.")
        
        for pert_type, intensities in self.merge_config.base_perturbation_intensities.items():
            for intensity in intensities:
                processed_config = self._create_processed_config(config, pert_type, intensity)
                
                if not self._processed_dataset_exists(processed_config):
                    logger.info(f"Generating processed dataset for {pert_type} with intensity {intensity}")
                    result = self.processor.process_dataset(processed_config)
                    self.processor.cache_dataset(result, processed_config)
    
    def merge_datasets(self, config: BaseDatasetConfig) -> DatasetResult:
        """Merge all processed datasets into a single dataset"""
        logger.info(f"Merging datasets.")
        
        # Ensure all processed datasets exist
        self.ensure_processed_datasets(config)
        config.num_perturbation_types = self.merge_config.num_perturbation_types
        
        # Load and merge all processed datasets
        merged_df = pd.DataFrame()
        
        for pert_type, intensities in self.merge_config.base_perturbation_intensities.items():
            for intensity in intensities:
                processed_config = self._create_processed_config(config, pert_type, intensity)
                result = self.processor.process_dataset(processed_config)
                
                df = self._convert_result_to_dataframe(result)
                df['perturbation_type'] = pert_type.value
                df['perturbation_intensity'] = intensity
                
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        
        # Save merged dataset
        merged_path = self._get_merged_path(config)
        os.makedirs(os.path.dirname(merged_path), exist_ok=True)
        merged_df.to_csv(merged_path, index=False)
        
        return self._convert_dataframe_to_result(merged_df, config)
    
    def _processed_dataset_exists(self, config: BaseDatasetConfig) -> bool:
        """Check if processed dataset exists in cache"""
        cache_dir = self.processor.file_handler.get_dataset_dir(config.source_type)
        cache_path = self.processor.file_handler.get_cache_path(cache_dir, config)
        return os.path.exists(cache_path)

    def _create_processed_config(
        self,
        base_config: BaseDatasetConfig,
        pert_type: PerturbationType,
        intensity: int
    ) -> BaseDatasetConfig:
        """Create config for processed dataset with specific perturbation."""
        return replace(
            base_config,
            source_type=DatasetSourceType.PROCESSED,
            perturbation_type=pert_type,
            perturbation_intensity=intensity,
        )
    
    def _get_merged_path(self, config: BaseDatasetConfig) -> str:
        """Get path for merged dataset"""
        cache_dir = self.processor.file_handler.get_dataset_dir(DatasetSourceType.MERGED)
        return self.processor.file_handler.get_cache_path(cache_dir, config)
    
    def _convert_result_to_dataframe(self, result: DatasetResult) -> pd.DataFrame:
        """Convert dataset result to DataFrame"""
        data = []
        for entry in result.entries:
            row = {
                'question': entry.question,
                'answer': entry.answer,
                **entry.metadata
            }
            data.append(row)
        return pd.DataFrame(data)
    
    def _convert_dataframe_to_result(self, df: pd.DataFrame, config: BaseDatasetConfig) -> DatasetResult:
        """Convert DataFrame back to DatasetResult"""
        entries = []
        perturbation_configs = []
        
        for _, row in df.iterrows():
            entry = DatasetEntry(
                question=row['question'],
                answer=row['answer'],
                metadata={k: v for k, v in row.items() if k not in ['question', 'answer']}
            )
            entries.append(entry)
            
            if 'perturbation_type' in row and 'perturbation_intensity' in row:
                pert_config = PerturbationConfig(
                    type=row['perturbation_type'],
                    intensity=row['perturbation_intensity']
                )
                perturbation_configs.append(pert_config)
        
        return DatasetResult(
            entries=entries,
            perturbation_config=perturbation_configs if perturbation_configs else None,
            config=config
        )
