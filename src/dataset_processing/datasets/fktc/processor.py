import os
import pandas as pd

from pandas.errors import EmptyDataError
from typing import Optional

from src.config import DATA_DIR
from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.merge_config import MergeConfig
from src.dataset_processing.common.constants import DATA_FILES
from src.dataset_processing.common.field_names import DatasetEntryNames, PerturbationNames
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.common.dataset_result import DatasetResult
from src.dataset_processing.common.base_processor import DatasetProcessor
from src.dataset_processing.datasets.fktc.data_processor import FKTCDataProcessor
from src.dataset_processing.datasets.fktc.file_handler import FKTCFileHandler
from src.dataset_processing.datasets.fktc.models import ProcessedFKTCEntry
from src.dataset_processing.dataset_merger import DatasetMerger
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
import logging

logger = logging.getLogger("reliability_eval")

class FKTCProcessor(DatasetProcessor):
    """Processor for FKTC datasets"""
    def __init__(
        self,
        base_dir: str = DATA_DIR,
        merge_config: Optional[MergeConfig] = None
    ):
        super().__init__()
        self.base_dir = base_dir
        self.file_handler = FKTCFileHandler(base_dir)
        self.data_processor = FKTCDataProcessor()
        self.merger = DatasetMerger(
            processor=self,
            merge_config=merge_config
        )
        
    def load_from_cache(self, config: BaseDatasetConfig) -> Optional[DatasetResult]:
        """Load dataset from cache if available"""
        cache_dir = self.file_handler.get_dataset_dir(config.source_type)
        cache_path = self.file_handler.get_cache_path(cache_dir, config)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            df = pd.read_csv(cache_path)
            if config.num_entries:
                if config.source_type == DatasetSourceType.MERGED:
                    df = df.head(config.num_entries * config.num_relations * config.num_perturbation_types)
                else:
                    df = df.head(config.num_entries * config.num_relations)
            
            return self._create_result_from_dataframe(df)
        except Exception as e:
            logger.error(f"Error loading cached dataset: {str(e)}")
            return None
    
    def process_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process FKTC dataset according to configuration"""
        if config.dataset_name not in DATA_FILES:
            raise ValueError(f"Invalid dataset name: {config.dataset_name}")
        
        if config.source_type == DatasetSourceType.MERGED:
            try:
                return self._process_merged_dataset(config)
            except FileNotFoundError:
                logger.info(f"Merged dataset not found, triggering merge operation")
                return self.merger.merge_datasets(config)
        elif config.source_type == DatasetSourceType.PROCESSED:
            if config.force_reprocess:
                # Process from raw and save to processed directory
                result = self._process_raw_dataset(config)
                self._save_to_processed(result, config)
                return result
            # Try loading from processed directory
            try:
                return self._load_processed_dataset(config)
            except FileNotFoundError:
                # If processed file doesn't exist, process from raw
                result = self._process_raw_dataset(config)
                self._save_to_processed(result, config)
                return result
        else:  # RAW
            return self._process_raw_dataset(config)
    
    def _process_raw_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process raw FKTC dataset"""
        df = self.file_handler.read_csv_file(config.dataset_name, DatasetSourceType.RAW)
        
        # Get few-shot examples if needed
        few_shot_examples = None
        if hasattr(config, 'num_shots') and config.num_shots > 0:
            few_shot_examples = self.data_processor.get_few_shot_examples(df, config.num_shots)
            # Skip the examples used for few-shot prompts
            df = df.iloc[config.num_shots:]
        
        if config.num_entries:
            df = df.sample(n=config.num_entries, random_state=config.random_seed)
        
        processed_entries = []
        for _, row in df.iterrows():
            entry = self.data_processor.parse_raw_entry({
                'subject': row['subject'],
                'object': row['answer'],
                'taxonomy': row['taxonomy'].split(',') if row['taxonomy'] else []
            })
            
            perturbation_config = None
            if config.source_type == DatasetSourceType.PROCESSED:
                perturbation_config = PerturbationConfig(
                    type=config.perturbation_type,
                    intensity=config.perturbation_intensity
                )
            
            processed_entry = self.data_processor.create_processed_entry(
                entry,
                row['relation'],
                perturbation_config,
                few_shot_examples=few_shot_examples
            )
            processed_entries.append(self._convert_to_dataset_entry(processed_entry))
        
        return self._create_dataset_result(processed_entries, config)
    
    def _load_from_csv(self, file_path: str, config: BaseDatasetConfig) -> DatasetResult:
        """Load and process dataset from CSV"""
        try:
            df = pd.read_csv(file_path)
        except EmptyDataError:
            raise FileNotFoundError(f"Empty dataset file: {file_path}")
        
        if config.num_entries:
            df = df.head(config.num_entries * config.num_relations)
            
        entries = []
        perturbation_config = []
        
        for _, row in df.iterrows():
            entries.append(self._create_dataset_entry_from_row(row))
            if PerturbationNames.PERTURBATION_TYPE.value in row and PerturbationNames.PERTURBATION_INTENSITY.value in row:
                perturbation_config.append(
                    PerturbationConfig(
                        type=row[PerturbationNames.PERTURBATION_TYPE.value],
                        intensity=row[PerturbationNames.PERTURBATION_INTENSITY.value],
                        taxonomies=row.get('taxonomy', None)
                    )
                )

        return DatasetResult(
            entries=entries,
            perturbation_config=perturbation_config if perturbation_config else None,
            config=config
        )
    
    def _convert_to_dataset_entry(self, processed_entry: ProcessedFKTCEntry) -> DatasetEntry:
        """Convert processed entry to dataset entry"""
        return DatasetEntry(
            question=processed_entry.question,
            answer=processed_entry.answer,
            metadata=processed_entry.metadata
        )
    
    def _create_dataset_entry_from_row(self, row: pd.Series) -> DatasetEntry:
        """Create dataset entry from DataFrame row"""
        metadata = {
            key: row[key] for key in row.keys()
            if key not in [DatasetEntryNames.QUESTION.value, DatasetEntryNames.ANSWER.value, PerturbationNames.PERTURBATION_TYPE.value, PerturbationNames.PERTURBATION_INTENSITY.value]
        }
        return DatasetEntry(
            question=row[DatasetEntryNames.QUESTION.value],
            answer=row[DatasetEntryNames.ANSWER.value],
            metadata=metadata
        )
    
