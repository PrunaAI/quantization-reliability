from src.dataset_processing.perplexity.common.processor import SimpleTextPerplexityProcessor
from src.dataset_processing.perplexity.datasets.c4.loader import C4Loader


class C4Processor(SimpleTextPerplexityProcessor):
    """Processes C4 dataset for perplexity evaluation."""
    loader_class = C4Loader
