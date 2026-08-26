import os
import random
from typing import List, Optional

import logging
import pandas as pd

from pandas.errors import EmptyDataError

from src.config import DATA_DIR
from src.dataset_processing.common.base_configs import BaseDatasetConfig
from src.dataset_processing.common.merge_config import MergeConfig
from src.dataset_processing.common.field_names import PerturbationNames
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.common.dataset_entry import DatasetEntry
from src.dataset_processing.common.dataset_result import DatasetResult
from src.dataset_processing.common.base_processor import DatasetProcessor
from src.dataset_processing.datasets.commonsenseqa.data_processor import CommonSenseQADataProcessor
from src.dataset_processing.datasets.commonsenseqa.file_handler import CommonSenseQAFileHandler
from src.dataset_processing.datasets.commonsenseqa.models import ProcessedCommonSenseQAEntry
from src.dataset_processing.dataset_merger import DatasetMerger

from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

logger = logging.getLogger("reliability_eval")


class CommonSenseQAProcessor(DatasetProcessor):
    """Processor for CommonSenseQA datasets."""
    
    def __init__(
        self,
        base_dir: str = DATA_DIR,
        merge_config: Optional[MergeConfig] = None
    ):
        """Initialize processor with base directory."""
        super().__init__()
        self.base_dir = base_dir
        self.file_handler = CommonSenseQAFileHandler(base_dir)
        self.data_processor = CommonSenseQADataProcessor()
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
        """Process CommonSenseQA dataset according to configuration."""
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
    
    def _process_raw_dataset(self, config: BaseDatasetConfig) -> DatasetResult:
        """Process raw CommonSenseQA dataset."""
        # Create a fixed random state for consistent sampling
        local_random = random.Random(config.random_seed)
        
        # Check for cached raw dataset
        raw_dir = self.file_handler.get_dataset_dir(DatasetSourceType.RAW)
        raw_path = self.file_handler.get_cache_path(raw_dir, config)
        
        if not config.force_reprocess and os.path.exists(raw_path):
            return self._load_from_csv(raw_path, config)
            
        # If not cached or force_reprocess, load from HuggingFace
        dataset = self.file_handler.load_huggingface_dataset(config.split)
        processed_entries = []
        
        # Convert dataset to list and sample if needed
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
            
            # Add this section to handle few-shot examples
            few_shot_examples = None
            if hasattr(config, 'num_shots') and config.num_shots > 0:
                few_shot_examples = self.data_processor.get_few_shot_examples(config.num_shots)
            
            processed_entry = self.data_processor.create_processed_entry(
                entry,
                perturbation_config=perturbation_config,
                few_shot_examples=few_shot_examples,
                num_shots=config.num_shots if hasattr(config, 'num_shots') else 0
            )
            
            processed_entries.append(self._convert_to_dataset_entry(processed_entry))
        
        result = self._create_dataset_result(processed_entries, config)
        
        # Cache raw dataset if it's the raw source type
        if config.source_type == DatasetSourceType.RAW:
            os.makedirs(raw_dir, exist_ok=True)
            df = self._convert_result_to_dataframe(result)
            df.to_csv(raw_path, index=False)
            logger.info(f"Cached raw dataset to {raw_path}")
        
        return result
    
    def _load_from_csv(self, file_path: str, config: BaseDatasetConfig) -> DatasetResult:
        """Load and process dataset from CSV."""
        try:
            df = pd.read_csv(file_path)
        except EmptyDataError:
            raise FileNotFoundError(f"Empty dataset file: {file_path}")
        
        if config.num_entries:
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
        """Create dataset entry from DataFrame row preserving structure through metadata."""
        # Extract choices from columns
        choices = [row[f'choice_{i}'] for i in range(5)]
        
        # Create metadata including structured components
        metadata = {
            'raw_question': row['raw_question'],
            'choices': choices,
            **{k: v for k, v in row.items() 
            if not k.startswith('choice_') 
            and k not in ['question', 'answer', 'raw_question', 'perturbation_type', 'perturbation_intensity']}
        }
        
        return DatasetEntry(
            question=row['question'],
            answer=row['answer'],
            metadata=metadata
        )
    
    def _convert_to_dataset_entry(self, processed_entry: ProcessedCommonSenseQAEntry) -> DatasetEntry:
        """Convert processed entry to dataset entry storing structure in metadata."""
        # Parse the formatted question back into components
        lines = processed_entry.question.split('\n')
        question_text = '\n'.join(lines[:-6])
        choices = [line.split('. ')[1] for line in lines[-6:-1]]
        
        # Store original components in metadata
        metadata = {
            **processed_entry.metadata,
            'raw_question': question_text,
            'choices': choices
        }
        
        return DatasetEntry(
            question=self._format_question(question_text, choices),
            answer=processed_entry.answer,
            metadata=metadata
        )
    
    def _format_question(self, question_text: str, choices: List[str]) -> str:
        """Format question consistently with choices."""
        formatted = f"{question_text.strip()}\n"
        for idx, choice in enumerate(choices):
            letter = chr(65 + idx)  # A, B, C, D, E
            formatted += f"{letter}. {choice}\n"
        formatted += "Answer:"
        return formatted
    
    def _convert_result_to_dataframe(self, result: DatasetResult) -> pd.DataFrame:
        """Convert dataset result to DataFrame preserving structure in metadata columns."""
        data = []
        for entry in result.entries:
            row = {
                'question': entry.question,
                'answer': entry.answer,
                'raw_question': entry.metadata.get('raw_question', ''),
                **{f'choice_{i}': choice for i, choice in enumerate(entry.metadata.get('choices', []))},
                **{k: v for k, v in entry.metadata.items() if k not in ['raw_question', 'choices']}
            }
            if result.perturbation_config:
                row.update({
                    'perturbation_type': result.perturbation_config[0].type,
                    'perturbation_intensity': result.perturbation_config[0].intensity
                })
            data.append(row)
        return pd.DataFrame(data)
