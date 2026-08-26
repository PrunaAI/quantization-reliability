from dataclasses import dataclass
import re
from typing import Optional


@dataclass
class WordProcessor:
    """Processor for word-level operations."""
    
    @staticmethod
    def extract_punctuation(word: str) -> tuple[str, Optional[str], Optional[str]]:
        """Extract punctuation from word boundaries."""
        start_punct = re.match(r'^[^\w\s]+', word)
        end_punct = re.search(r'[^\w\s]+$', word)
        clean_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
        return clean_word, start_punct, end_punct
    
    @staticmethod
    def restore_punctuation(word: str, start_punct: Optional[str], end_punct: Optional[str]) -> str:
        """Restore punctuation to word boundaries."""
        result = word
        if start_punct:
            result = start_punct.group() + result
        if end_punct:
            result = result + end_punct.group()
        return result