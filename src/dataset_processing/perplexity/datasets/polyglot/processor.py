from typing import List, Tuple
from src.dataset_processing.perplexity.common.config.base_configs import PerplexityDatasetConfig
from src.dataset_processing.perplexity.common.models.dataset_entry import PerplexityDatasetEntry
from src.dataset_processing.perplexity.common.processor import BaseProcessor
from src.dataset_processing.perplexity.common.tokenization import PerplexityTokenizer
from src.dataset_processing.perplexity.datasets.polyglot.loader import PolyglotLoader


class PolyglotProcessor(BaseProcessor):
    """Processes Polyglot dataset for perplexity evaluation."""
    def load_raw_data(self, config: PerplexityDatasetConfig) -> Tuple[List[str], List[str]]:
        return PolyglotLoader.load_raw_data(
            split=config.split,
            n_samples=config.n_samples
        )

    def process_raw_data(self, raw_data: Tuple[List[str], List[str]], config: PerplexityDatasetConfig) -> List[PerplexityDatasetEntry]:
        questions, answers = raw_data
        tokenizer = PerplexityTokenizer(self.tokenizer)
        return tokenizer.tokenize_qa_pairs(questions, answers, config.seq_length)
