from src.dataset_processing.perplexity.common.processor import SimpleTextPerplexityProcessor
from src.dataset_processing.perplexity.datasets.ptb.loader import PTBLoader


class PTBProcessor(SimpleTextPerplexityProcessor):
    """Processes PTB dataset for perplexity evaluation."""
    loader_class = PTBLoader
