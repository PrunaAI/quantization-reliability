import os
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.field_names import DatasetEntryNames, PerturbationNames
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.common.dataset_result import DatasetResult
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
import logging

logger = logging.getLogger("reliability_eval")


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors"""

    @abstractmethod
    def process_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process dataset according to configuration"""
        pass

    @abstractmethod
    def load_from_cache(self, config: BaseDatasetConfig) -> Optional[DatasetResult]:
        """Load dataset from cache if available"""
        pass

    def cache_dataset(self, result: DatasetResult, config: BaseDatasetConfig) -> None:
        """Cache processed dataset."""
        cache_dir = self.file_handler.get_dataset_dir(config.source_type)
        cache_path = self.file_handler.get_cache_path(cache_dir, config)

        df = self._convert_result_to_dataframe(result)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached dataset to {cache_path}")

    def _load_processed_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Load already processed dataset."""
        processed_dir = self.file_handler.get_dataset_dir(DatasetSourceType.PROCESSED)
        processed_path = self.file_handler.get_cache_path(processed_dir, config)

        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

        return self._load_from_csv(processed_path, config)

    def _save_to_processed(self, result: DatasetResult, config: BaseDatasetConfig) -> None:
        """Save processed dataset to the processed directory."""
        processed_dir = self.file_handler.get_dataset_dir(DatasetSourceType.PROCESSED)
        processed_path = self.file_handler.get_cache_path(processed_dir, config)

        os.makedirs(processed_dir, exist_ok=True)

        df = self._convert_result_to_dataframe(result)
        df.to_csv(processed_path, index=False)
        logger.info(f"Saved processed dataset to {processed_path}")

    def _process_merged_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process merged dataset."""
        merged_dir = self.file_handler.get_dataset_dir(DatasetSourceType.MERGED)
        merged_path = self.file_handler.get_cache_path(merged_dir, config)

        if not os.path.exists(merged_path):
            raise FileNotFoundError(f"Merged dataset not found: {merged_path}")

        return self._load_from_csv(merged_path, config)

    def _create_dataset_result(
        self,
        entries: List[DatasetEntry],
        config: BaseDatasetConfig
    ) -> DatasetResult:
        """Create dataset result with perturbation info."""
        return DatasetResult(
            entries=entries,
            perturbation_config=[PerturbationConfig(config.perturbation_type, config.perturbation_intensity)]
            if config.source_type == DatasetSourceType.PROCESSED and config.perturbation_type != "none" else None,
            config=config
        )

    def _convert_result_to_dataframe(self, result: DatasetResult) -> pd.DataFrame:
        """Convert dataset result to DataFrame."""
        data = []
        for entry in result.entries:
            row = {
                DatasetEntryNames.QUESTION.value: entry.question,
                DatasetEntryNames.ANSWER.value: entry.answer,
                **entry.metadata
            }
            if result.perturbation_config:
                row.update({
                    PerturbationNames.PERTURBATION_TYPE.value: result.perturbation_config[0].type,
                    PerturbationNames.PERTURBATION_INTENSITY.value: result.perturbation_config[0].intensity
                })
            data.append(row)
        return pd.DataFrame(data)

    def _create_result_from_dataframe(self, df: pd.DataFrame) -> DatasetResult:
        """Create dataset result from DataFrame."""
        entries = []
        perturbation_config = []

        for _, row in df.iterrows():
            entries.append(self._create_dataset_entry_from_row(row))
            if PerturbationNames.PERTURBATION_TYPE.value in row and PerturbationNames.PERTURBATION_INTENSITY.value in row:
                perturbation_config.append(
                    PerturbationConfig(
                        type=row[PerturbationNames.PERTURBATION_TYPE.value],
                        intensity=row[PerturbationNames.PERTURBATION_INTENSITY.value]
                    )
                )

        return DatasetResult(
            entries=entries,
            perturbation_config=perturbation_config if perturbation_config else None,
            config=None
        )
