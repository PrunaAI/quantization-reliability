from typing import List
import random

from src.dataset_processing.perturbations.base.char_perturbation import CharacterPerturbation
from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

class CharacterDeleter:
    """Handles character deletion operations on a text string."""
    
    def get_random_positions(self, text_length: int, num_deletions: int) -> List[int]:
        """Get random positions for character deletions."""
        if text_length <= 0 or num_deletions <= 0:
            return []
        positions = random.sample(range(text_length), min(num_deletions, text_length))
        return sorted(positions, reverse=True)
    
    def delete_characters(self, text: str, positions: List[int]) -> str:
        """Delete characters at specified positions."""
        result = list(text)
        for pos in positions:
            del result[pos]
        return ''.join(result)

class CharDeletion(TextPerturbation):
    """Implements random character deletion across entire text."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.deleter = CharacterDeleter()
    
    def perturb(self, text: str) -> str:
        """Delete random characters from the question part of the text."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.deleter.get_random_positions(
            len(question_part),
            self.config.intensity
        )
        
        perturbed_question = self.deleter.delete_characters(question_part, positions)
        return perturbed_question + answer_part