import os
import random
from typing import Optional
import pandas as pd

from pandas.errors import EmptyDataError

from src.config import DATA_DIR
from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.merge_config import MergeConfig
from src.dataset_processing.common.field_names import DatasetEntryNames, PerturbationNames
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.common.dataset_result import DatasetResult
from src.dataset_processing.common.base_processor import DatasetProcessor
from src.dataset_processing.datasets.coqa.data_processor import CoQADataProcessor
from src.dataset_processing.datasets.coqa.file_handler import CoQAFileHandler
from src.dataset_processing.datasets.coqa.models import ProcessedCoQAEntry
from src.dataset_processing.dataset_merger import DatasetMerger
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
import logging

logger = logging.getLogger("reliability_eval")

class CoQAProcessor(DatasetProcessor):
    """Processor for CoQA datasets."""
    
    def __init__(
        self,
        base_dir: str = DATA_DIR,
        merge_config: Optional[MergeConfig] = None
    ):
        super().__init__()
        self.base_dir = base_dir
        self.file_handler = CoQAFileHandler(base_dir)
        self.data_processor = CoQADataProcessor()
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
                if config.source_type == DatasetSourceType.MERGED:
                    df = df.head(config.num_entries * config.num_perturbation_types)
                else:
                    df = df.head(config.num_entries)
                
            return self._create_result_from_dataframe(df)
        except Exception as e:
            logger.error(f"Error loading cached dataset: {str(e)}")
            return None
    
    def process_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process CoQA dataset according to configuration."""
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
            return self._process_raw_dataset(config)
    
    def _load_from_csv(self, file_path: str, config: BaseDatasetConfig) -> DatasetResult:
        """Load and process dataset from CSV."""
        try:
            df = pd.read_csv(file_path)
        except EmptyDataError:
            raise FileNotFoundError(f"Empty dataset file: {file_path}")
        
        if config.num_entries:
            df = df.head(config.num_entries)  # No relations multiplier needed for CoQA
            
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
        """Create dataset entry from DataFrame row for CoQA."""
        metadata = {
            key: row[key] for key in row.keys()
            if key not in [
                DatasetEntryNames.QUESTION.value,
                DatasetEntryNames.ANSWER.value,
                PerturbationNames.PERTURBATION_TYPE.value,
                PerturbationNames.PERTURBATION_INTENSITY.value
            ] and key in [
                'story_id',
                'question_id',
                'original_question',
                'qa_history'  # Include qa_history if present
            ]
        }
        
        return DatasetEntry(
            question=row[DatasetEntryNames.QUESTION.value],
            answer=row[DatasetEntryNames.ANSWER.value],
            metadata=metadata
        )
        
    def _process_raw_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process raw CoQA dataset."""
        # Create a fixed random state for consistent sampling
        local_random = random.Random(config.random_seed)

        raw_data = self.file_handler.read_json_file(source_type=DatasetSourceType.RAW, split=config.split)['data']
        
        # Create list of all possible (story, question_idx) pairs
        all_qa_pairs = [(sample, q_idx) 
                        for sample in raw_data 
                        for q_idx in range(min(len(sample['questions']), 
                                            config.questions_per_conversation if config.questions_per_conversation else len(sample['questions'])))]
            
        # Sample if needed
        if config.num_entries:
            # Use local_random instead of global random
            all_qa_pairs = local_random.sample(all_qa_pairs, config.num_entries)

        processed_entries = []
        for sample, question_index in all_qa_pairs:
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']
            
            # Build QA history up to current question
            qa_history = ""
            for prev_q, prev_a in zip(questions[:question_index], answers[:question_index]):
                qa_history += f"\n\nQ: {prev_q['input_text']}\nA: {prev_a['input_text']}"
            
            entry = self.data_processor.parse_raw_entry(
                story=story,
                question=questions[question_index],
                answer=answers[question_index],
                story_id=sample.get('id', '')
            )
            
            perturbation_config = None
            if config.source_type == DatasetSourceType.PROCESSED:
                perturbation_config = PerturbationConfig(
                    type=config.perturbation_type,
                    intensity=config.perturbation_intensity
                )
                
            processed_entry = self.data_processor.create_processed_entry(
                entry,
                qa_history,
                perturbation_config
            )
            processed_entries.append(self._convert_to_dataset_entry(processed_entry))

        return self._create_dataset_result(processed_entries, config)
    
    def _convert_to_dataset_entry(self, processed_entry: ProcessedCoQAEntry) -> DatasetEntry:
        """Convert processed entry to dataset entry."""
        return DatasetEntry(
            question=processed_entry.question,
            answer=processed_entry.answer,
            metadata=processed_entry.metadata
        )
    
    def _create_result_from_dataframe(self, df: pd.DataFrame) -> DatasetResult:
        """Create dataset result from DataFrame."""
        entries = []
        perturbation_config = []
        
        for _, row in df.iterrows():
            metadata = {
                key: row[key] for key in row.keys()
                if key not in [DatasetEntryNames.QUESTION.value, DatasetEntryNames.ANSWER.value, 
                             PerturbationNames.PERTURBATION_TYPE.value, PerturbationNames.PERTURBATION_INTENSITY.value]
            }
            
            entries.append(DatasetEntry(
                question=row[DatasetEntryNames.QUESTION.value],
                answer=row[DatasetEntryNames.ANSWER.value],
                metadata=metadata
            ))
            
            if (PerturbationNames.PERTURBATION_TYPE.value in row and 
                PerturbationNames.PERTURBATION_INTENSITY.value in row):
                perturbation_config.append(PerturbationConfig(
                    type=row[PerturbationNames.PERTURBATION_TYPE.value],
                    intensity=row[PerturbationNames.PERTURBATION_INTENSITY.value]
                ))
        
        return DatasetResult(
            entries=entries,
            perturbation_config=perturbation_config if perturbation_config else None,
            config=None
        )
