import logging
import os
import random
import pandas as pd

from typing import Optional
from pandas.errors import EmptyDataError

from src.config import DATA_DIR
from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.merge_config import MergeConfig
from src.dataset_processing.common.field_names import DatasetEntryNames, PerturbationNames
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.common.dataset_result import DatasetResult
from src.dataset_processing.common.base_processor import DatasetProcessor
from src.dataset_processing.datasets.triviaqa.data_processor import TriviaQADataProcessor
from src.dataset_processing.datasets.triviaqa.file_handler import TriviaQAFileHandler
from src.dataset_processing.datasets.triviaqa.models import ProcessedTriviaQAEntry
from src.dataset_processing.dataset_merger import DatasetMerger
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

logger = logging.getLogger("reliability_eval")


class TriviaQAProcessor(DatasetProcessor):
    """Processor for TriviaQA datasets."""
    
    def __init__(
        self,
        base_dir: str = DATA_DIR,
        merge_config: Optional[MergeConfig] = None
    ):
        """Initialize processor with base directory."""
        super().__init__()
        self.base_dir = base_dir
        self.file_handler = TriviaQAFileHandler(base_dir)
        self.data_processor = TriviaQADataProcessor()
        self.merger = DatasetMerger(
            processor=self,
            merge_config=merge_config
        )
    
    def load_from_cache(self, config: BaseDatasetConfig) -> Optional[DatasetResult]:
        """Load dataset from cache if available."""
        cache_dir = self.file_handler.get_dataset_dir(config.source_type)
        cache_path = self.file_handler.get_cache_path(cache_dir, config)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            df = pd.read_csv(cache_path)
            if config.num_entries:
                df = df.head(config.num_entries)
                
            return self._create_result_from_dataframe(df)
        except Exception as e:
            logger.error(f"Error loading cached dataset: {str(e)}")
            return None
    
    def process_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process TriviaQA dataset according to configuration."""
        if config.source_type == DatasetSourceType.MERGED:
            try:
                return self._process_merged_dataset(config)
            except FileNotFoundError:
                logger.info("Merged dataset not found, triggering merge operation")
                return self.merger.merge_datasets(config)
        elif config.source_type == DatasetSourceType.PROCESSED:
            if config.force_reprocess:
                result = self._process_raw_dataset(config)
                self._save_to_processed(result, config)
                return result
            try:
                return self._load_processed_dataset(config)
            except FileNotFoundError:
                result = self._process_raw_dataset(config)
                self._save_to_processed(result, config)
                return result
        else:  # RAW
            if config.force_reprocess:
                result = self._process_raw_dataset(config)
                self._save_to_raw(result, config)
                return result
            try:
                return self._load_raw_dataset(config)  # New method needed
            except FileNotFoundError:
                result = self._process_raw_dataset(config)
                self._save_to_raw(result, config)
                return result
        
    def _process_raw_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process raw TriviaQA dataset."""
        # Create a fixed random state for consistent sampling
        local_random = random.Random(config.random_seed)
        
        dataset = self.file_handler.load_huggingface_dataset(config.version, config.split)
        
        # Get few-shot examples if needed
        few_shot_examples = None
        if hasattr(config, 'num_shots') and config.num_shots > 0:
            few_shot_examples = self.data_processor.get_few_shot_examples(config.num_shots)
            # Skip the examples used for few-shot prompts in the actual dataset processing
            dataset = dataset.select(range(config.num_shots, len(dataset)))
        
        processed_entries = []
        raw_entries = list(dataset)
        if config.num_entries:
            # Use local_random instead of global random
            raw_entries = local_random.sample(raw_entries, config.num_entries)

        for raw_entry in raw_entries:
            entry = self.data_processor.parse_raw_entry(raw_entry)
            
            perturbation_config = None
            if config.source_type == DatasetSourceType.PROCESSED:
                perturbation_config = PerturbationConfig(
                    type=config.perturbation_type,
                    intensity=config.perturbation_intensity
                )
            
            processed_entry = self.data_processor.create_processed_entry(
                entry,
                perturbation_config=perturbation_config,
                few_shot_examples=few_shot_examples
            )
            
            processed_entries.append(self._convert_to_dataset_entry(processed_entry))
        
        return self._create_dataset_result(processed_entries, config)
    
    def _save_to_raw(self, result: DatasetResult, config: BaseDatasetConfig) -> None:
        """Save raw dataset to the raw directory."""
        raw_dir = self.file_handler.get_dataset_dir(DatasetSourceType.RAW)
        raw_path = self.file_handler.get_cache_path(raw_dir, config)
        
        os.makedirs(raw_dir, exist_ok=True)
        
        df = self._convert_result_to_dataframe(result)
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw dataset to {raw_path}")

    def _load_raw_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Load already processed raw dataset."""
        raw_dir = self.file_handler.get_dataset_dir(DatasetSourceType.RAW)
        raw_path = self.file_handler.get_cache_path(raw_dir, config)
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
        
        return self._load_from_csv(raw_path, config)

    def _load_from_csv(self, file_path: str, config: BaseDatasetConfig) -> DatasetResult:
        """Load and process dataset from CSV."""
        try:
            df = pd.read_csv(file_path)
        except EmptyDataError:
            raise FileNotFoundError(f"Empty dataset file: {file_path}")
        
        if config.num_entries:
            if config.source_type == DatasetSourceType.MERGED:
                df = df.head(config.num_entries * config.num_perturbation_types)
            else:
                df = df.head(config.num_entries)
            
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
            config=config
        )
    
    def _create_dataset_entry_from_row(self, row: pd.Series) -> DatasetEntry:
        """Create dataset entry from DataFrame row."""
        metadata = {
            key: row[key] for key in row.keys()
            if key not in [
                DatasetEntryNames.QUESTION.value,
                DatasetEntryNames.ANSWER.value,
                PerturbationNames.PERTURBATION_TYPE.value,
                PerturbationNames.PERTURBATION_INTENSITY.value
            ]
        }
        
        # Convert answer string back to list if needed
        if isinstance(row[DatasetEntryNames.ANSWER.value], str) and '|' in row[DatasetEntryNames.ANSWER.value]:
            answer_list = row[DatasetEntryNames.ANSWER.value].split('|')
            metadata['answer_list'] = answer_list
            answer = answer_list[0]  # Use first answer as primary
        else:
            answer = row[DatasetEntryNames.ANSWER.value]
        
        return DatasetEntry(
            question=row[DatasetEntryNames.QUESTION.value],
            answer=answer,
            metadata=metadata
        )
    
    def _convert_to_dataset_entry(self, processed_entry: ProcessedTriviaQAEntry) -> DatasetEntry:
        """Convert processed entry to dataset entry."""
        return DatasetEntry(
            question=processed_entry.question,
            answer=processed_entry.answer.split('|')[0],  # Use first answer as primary
            metadata=processed_entry.metadata
        )
    
