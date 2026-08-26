from typing import List
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

class CharacterSwapper:
    """Handles character swapping operations on a text string."""
    
    def get_random_positions(self, text_length: int, num_swaps: int) -> List[int]:
        """Get random positions for character swaps."""
        if text_length <= 1 or num_swaps <= 0:
            return []
        # Only select positions that have a next character to swap with
        available_positions = list(range(text_length - 1))
        return random.sample(available_positions, min(num_swaps, len(available_positions)))
    
    def swap_characters(self, text: str, positions: List[int]) -> str:
        """Swap adjacent characters at specified positions."""
        result = list(text)
        # Sort positions to avoid conflicts when swapping
        for pos in sorted(positions):
            result[pos], result[pos + 1] = result[pos + 1], result[pos]
        return ''.join(result)

class CharSwapping(TextPerturbation):
    """Implements random adjacent character swapping across entire text."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.swapper = CharacterSwapper()
    
    def perturb(self, text: str) -> str:
        """Swap random adjacent characters in the question part of the text."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.swapper.get_random_positions(
            len(question_part),
            self.config.intensity
        )
        
        perturbed_question = self.swapper.swap_characters(question_part, positions)
        return perturbed_question + answer_part