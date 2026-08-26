from src.dataset_processing.perplexity.common.processor import SimpleTextPerplexityProcessor
from src.dataset_processing.perplexity.datasets.wikitext.loader import WikiTextLoader


class WikiTextProcessor(SimpleTextPerplexityProcessor):
    """Processes WikiText dataset for perplexity evaluation."""
    loader_class = WikiTextLoader
