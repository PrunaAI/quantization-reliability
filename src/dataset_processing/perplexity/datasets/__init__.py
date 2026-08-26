from .wikitext.processor import WikiTextProcessor
from .openassistant.processor import OpenAssistantProcessor
from .polyglot.processor import PolyglotProcessor
from .ptb.processor import PTBProcessor
from .c4.processor import C4Processor

__all__ = [
    "WikiTextProcessor",
    "OpenAssistantProcessor",
    "PolyglotProcessor",
    "PTBProcessor",
    "C4Processor"
]
