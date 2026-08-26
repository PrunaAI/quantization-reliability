from abc import abstractmethod
from typing import List, Optional

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation



class WordPerturbation(TextPerturbation):
    """Base class for word-level perturbations."""
    
    @abstractmethod
    def get_candidates(self, word: str) -> List[str]:
        """Get candidate replacements for a word."""
        pass

    @abstractmethod
    def select_replacement(self, word: str, candidates: List[str]) -> Optional[str]:
        """Select appropriate replacement from candidates."""
        pass