from typing import List, Tuple, Set
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.utils.word_processor import WordProcessor

class SwapManager:
    """Handler for word swapping operations."""
    def get_available_positions(self, length: int) -> Set[int]:
        """Get positions available for swapping."""
        return set(range(length - 1))

    def swap_words(self, words: List[str], position: int) -> None:
        """Swap words at given position with next word."""
        words[position], words[position + 1] = words[position + 1], words[position]

    def update_available_positions(self, positions: Set[int], used_position: int) -> None:
        """Update available positions after swap."""
        positions.discard(used_position)
        positions.discard(used_position + 1)

class WordSwapping(TextPerturbation):
    """Implementation of adjacent word swapping."""
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.word_processor = WordProcessor()
        self.swap_manager = SwapManager()

    def _get_swappable_pairs(self, length: int) -> List[Tuple[int, int]]:
        """Get pairs of indices that can be swapped."""
        return [(i, i + 1) for i in range(length - 1)]

    def _process_words(self, words: List[str]) -> List[Tuple[str, object, object]]:
        """Process words to handle punctuation."""
        return [self.word_processor.extract_punctuation(word) for word in words]

    def perturb(self, text: str) -> str:
        """Apply word swapping perturbation to text."""
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text

        # Process only question part
        words = question_part.split()
        if len(words) < 2:
            return text
            
        # Process words and maintain punctuation
        processed_words = self._process_words(words)
        clean_words = [word for word, _, _ in processed_words]
        
        # Get available swap pairs and perform swaps
        available_pairs = self._get_swappable_pairs(len(clean_words))
        num_swaps = min(self.config.intensity, len(available_pairs))
        
        if available_pairs and num_swaps > 0:
            chosen_pairs = random.sample(available_pairs, num_swaps)
            for pos1, pos2 in chosen_pairs:
                clean_words[pos1], clean_words[pos2] = clean_words[pos2], clean_words[pos1]
                processed_words[pos1], processed_words[pos2] = processed_words[pos2], processed_words[pos1]
        
        # Restore punctuation
        result_words = [
            self.word_processor.restore_punctuation(word, start_punct, end_punct)
            for word, start_punct, end_punct in processed_words
        ]
        
        # Combine processed question with unchanged answer part
        return " ".join(result_words) + answer_part
