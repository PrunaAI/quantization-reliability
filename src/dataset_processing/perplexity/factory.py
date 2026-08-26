from typing import Dict, Type
from transformers import PreTrainedTokenizer

from src.dataset_processing.perplexity.common.enums.dataset_types import PerplexityDatasetType
from src.dataset_processing.perplexity.common.processor import BaseProcessor
from src.dataset_processing.perplexity.datasets.c4.processor import C4Processor
from src.dataset_processing.perplexity.datasets.openassistant.processor import OpenAssistantProcessor
from src.dataset_processing.perplexity.datasets.polyglot.processor import PolyglotProcessor
from src.dataset_processing.perplexity.datasets.ptb.processor import PTBProcessor
from src.dataset_processing.perplexity.datasets.wikitext.processor import WikiTextProcessor

class PerplexityDatasetFactory:
    """Factory for creating perplexity dataset processors."""
    
    _processor_classes: Dict[PerplexityDatasetType, Type] = {
        PerplexityDatasetType.WIKITEXT: WikiTextProcessor,
        PerplexityDatasetType.OPENASSISTANT: OpenAssistantProcessor,
        PerplexityDatasetType.POLYGLOT: PolyglotProcessor,
        PerplexityDatasetType.PTB: PTBProcessor,
        PerplexityDatasetType.C4: C4Processor
    }

    @classmethod
    def create_processor(
        cls, 
        dataset_type: PerplexityDatasetType,
        tokenizer: PreTrainedTokenizer,
        stride: int = 512
    ) -> BaseProcessor:
        """Creates appropriate processor for dataset type."""
        if dataset_type not in cls._processor_classes:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        processor_class = cls._processor_classes[dataset_type]
        return processor_class(tokenizer=tokenizer, stride=stride)
