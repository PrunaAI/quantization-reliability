from typing import List

from src.dataset_processing.perplexity.common.chunking import DataChunker
from src.dataset_processing.perplexity.common.config.base_configs import PerplexityDatasetConfig
from src.dataset_processing.perplexity.common.models.dataset_entry import PerplexityDatasetEntry
from src.dataset_processing.perplexity.common.processor import BaseProcessor
from src.dataset_processing.perplexity.common.tokenization import PerplexityTokenizer
from src.dataset_processing.perplexity.datasets.openassistant.loader import OpenAssistantLoader
import logging


logger = logging.getLogger("reliability_eval")


class OpenAssistantProcessor(BaseProcessor):
    """Processes OpenAssistant dataset for perplexity evaluation."""
    def load_raw_data(self, config: PerplexityDatasetConfig) -> List[str]:
        if config.split not in ['validation']:
            logger.warning(f"OpenAssistant dataset does not have a '{config.split}' split. Using 'validation' instead.")
            config.split = 'validation'
            
        return OpenAssistantLoader.load_raw_data(
            split=config.split,
            n_samples=config.n_samples
        )

    def process_raw_data(self, raw_data: List[str], config: PerplexityDatasetConfig) -> List[PerplexityDatasetEntry]:
        tokenizer = PerplexityTokenizer(self.tokenizer)
        tokenized_data = tokenizer.tokenize_texts(raw_data)
        
        chunker = DataChunker(config.seq_length, self.stride)
        return list(chunker.create_chunks(tokenized_data))
