from typing import List
import random

from src.dataset_processing.perturbations.base.char_perturbation import CharacterPerturbation
from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

class CaseChanger:
    """Handles case changing operations on a text string."""
    
    def get_random_positions(self, text: str, num_changes: int) -> List[int]:
        """Get random positions of letters that can have their case changed."""
        changeable_positions = [i for i in range(len(text)) if text[i].isalpha()]
        if not changeable_positions or num_changes <= 0:
            return []
        return random.sample(changeable_positions, min(num_changes, len(changeable_positions)))
    
    def change_cases(self, text: str, positions: List[int]) -> str:
        """Change character cases at specified positions."""
        result = list(text)
        for pos in positions:
            result[pos] = result[pos].swapcase()
        return ''.join(result)

class CharCaseChange(TextPerturbation):
    """Implements random character case changes across entire text."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.case_changer = CaseChanger()
    
    def perturb(self, text: str) -> str:
        """Change cases of random letters in the question part of the text."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.case_changer.get_random_positions(
            question_part,
            self.config.intensity
        )
        
        perturbed_question = self.case_changer.change_cases(question_part, positions)
        return perturbed_question + answer_part
