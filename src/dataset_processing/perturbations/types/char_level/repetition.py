import random
from typing import List

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

class CharacterRepeater:
    """Handles character repetition operations on a text string."""
    
    def get_random_positions(self, text_length: int, num_repetitions: int) -> List[int]:
        """Get random positions for character repetitions."""
        if text_length <= 0 or num_repetitions <= 0:
            return []
        positions = random.sample(range(text_length), min(num_repetitions, text_length))
        return sorted(positions)  # Sort ascending since we're inserting
    
    def repeat_characters(self, text: str, positions: List[int]) -> str:
        """Repeat characters at specified positions."""
        result = list(text)
        offset = 0
        for pos in positions:
            result.insert(pos + offset, result[pos + offset])
            offset += 1
        return ''.join(result)

class CharRepetition(TextPerturbation):
    """Implements random character repetition across entire text."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.repeater = CharacterRepeater()
    
    def perturb(self, text: str) -> str:
        """Repeat random characters in the question part of the text."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.repeater.get_random_positions(
            len(question_part),
            self.config.intensity
        )
        
        perturbed_question = self.repeater.repeat_characters(question_part, positions)
        return perturbed_question + answer_part