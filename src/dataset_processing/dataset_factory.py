from typing import Type

from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.dataset_configs import CoQAConfig, CommonSenseQAConfig, FKTCConfig, MMLUConfig, TriviaQAConfig
from src.dataset_processing.common.merge_config import MergeConfig
from src.dataset_processing.common.dataset_types import DatasetType
from src.dataset_processing.common.base_processor import DatasetProcessor
from src.dataset_processing.datasets.commonsenseqa.processor import CommonSenseQAProcessor
from src.dataset_processing.datasets.coqa.processor import CoQAProcessor
from src.dataset_processing.datasets.fktc.processor import FKTCProcessor
from src.dataset_processing.datasets.mmlu.processor import MMLUProcessor
from src.dataset_processing.datasets.triviaqa.processor import TriviaQAProcessor


class DatasetFactory:
    """Factory for creating dataset processors"""

    @staticmethod
    def create_processor(
        dataset_type: DatasetType,
        merge_config: MergeConfig
    ) -> DatasetProcessor:
        """Create appropriate processor for dataset type"""
        processors = {
            DatasetType.FKTC: FKTCProcessor(merge_config=merge_config),
            DatasetType.COQA: CoQAProcessor(merge_config=merge_config),
            DatasetType.TRIVIAQA: TriviaQAProcessor(merge_config=merge_config),
            DatasetType.COMMONSENSEQA: CommonSenseQAProcessor(merge_config=merge_config),
            DatasetType.MMLU: MMLUProcessor(merge_config=merge_config)
        }
        return processors[dataset_type]

    @staticmethod
    def create_config(dataset_type: DatasetType) -> Type[BaseDatasetConfig]:
        """Create appropriate config class for dataset type. The same config class
        is used regardless of source type (raw/processed/merged) — pass `source_type`
        as a constructor argument."""
        configs = {
            DatasetType.FKTC: FKTCConfig,
            DatasetType.COQA: CoQAConfig,
            DatasetType.TRIVIAQA: TriviaQAConfig,
            DatasetType.COMMONSENSEQA: CommonSenseQAConfig,
            DatasetType.MMLU: MMLUConfig,
        }
        return configs[dataset_type]
