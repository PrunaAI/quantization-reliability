from typing import List, Set
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.constants import COMPLETE_STOP_WORDS
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.utils.word_processor import WordProcessor

class StopWordManager:
    """Handler for stop word operations."""
    
    def __init__(self):
        self.stop_words: Set[str] = set(COMPLETE_STOP_WORDS)
    
    def get_stop_word_indices(self, words: List[str]) -> List[int]:
        """Find indices of stop words in word list."""
        return [i for i, word in enumerate(words) if word.lower() in self.stop_words]

class WordDeletion(TextPerturbation):
    """Implementation of stop word deletion."""
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.word_processor = WordProcessor()
        self.stop_word_manager = StopWordManager()

    def _get_deletion_indices(self, words: List[str]) -> List[int]:
        """Get indices of words to be deleted."""
        stop_indices = [
            i for i, word in enumerate(words)
            if word.lower() in self.stop_word_manager.stop_words
        ]
        num_deletions = min(self.config.intensity, len(stop_indices))
        return random.sample(stop_indices, num_deletions) if stop_indices else []

    def perturb(self, text: str) -> str:
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text

        # Process only question part
        words = question_part.split()
        if not words:
            return text
            
        # Get indices to delete and remove words
        deletion_indices = self._get_deletion_indices(words)
        processed_words = [
            word for i, word in enumerate(words)
            if i not in deletion_indices
        ]

        # Combine processed question with unchanged answer part
        return " ".join(processed_words) + answer_part
