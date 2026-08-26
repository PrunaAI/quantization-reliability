from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import Any, List, Optional, Type

from src.dataset_processing.perplexity.common.chunking import DataChunker
from src.dataset_processing.perplexity.common.config.base_configs import PerplexityDatasetConfig
from src.dataset_processing.perplexity.common.dataset import BasePerplexityDataset
from src.dataset_processing.perplexity.common.models.dataset_entry import PerplexityDatasetEntry
from src.dataset_processing.perplexity.common.models.dataset_result import PerplexityDatasetResult
from src.dataset_processing.perplexity.common.tokenization import PerplexityTokenizer

class BaseProcessor(ABC):
    """Base processor for all perplexity datasets."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, stride: int = 512):
        self.tokenizer = tokenizer
        self.stride = stride

    @abstractmethod
    def load_raw_data(self, config: PerplexityDatasetConfig) -> Any:
        """Load dataset-specific raw data."""
        pass

    @abstractmethod
    def process_raw_data(self, raw_data: Any, config: PerplexityDatasetConfig) -> List[PerplexityDatasetEntry]:
        """Process dataset-specific raw data into entries."""
        pass

    def process_dataset(self, config: PerplexityDatasetConfig) -> DataLoader:
        """Common dataset processing pipeline."""
        raw_data = self.load_raw_data(config)
        entries = self.process_raw_data(raw_data, config)
        
        result = PerplexityDatasetResult(entries=entries, config=config)
        return DataLoader(
            BasePerplexityDataset(result.entries),
            batch_size=config.batch_size,
            shuffle=False
        )


class SimpleTextPerplexityProcessor(BaseProcessor):
    """Base processor for datasets whose loader returns a plain list of text samples."""

    loader_class: Optional[Type] = None

    def load_raw_data(self, config: PerplexityDatasetConfig) -> List[str]:
        return self.loader_class.load_raw_data(
            split=config.split,
            n_samples=config.n_samples
        )

    def process_raw_data(self, raw_data: List[str], config: PerplexityDatasetConfig) -> List[PerplexityDatasetEntry]:
        tokenizer = PerplexityTokenizer(self.tokenizer)
        tokenized_data = tokenizer.tokenize_texts(raw_data)

        chunker = DataChunker(config.seq_length, self.stride)
        return list(chunker.create_chunks(tokenized_data))