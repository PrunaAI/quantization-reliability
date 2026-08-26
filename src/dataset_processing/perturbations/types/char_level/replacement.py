import random

from dataclasses import dataclass
from typing import List, Dict
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

@dataclass
class ReplacementConfig:
    """Configuration for character replacement operations."""
    keyboard_layout: Dict[str, List[str]]
    case_sensitive: bool = True

class CharacterReplacer:
    """Handles character replacement operations on a text string."""
    
    def __init__(self, replacement_config: ReplacementConfig, case_sensitive: bool = True):
        """Initialize with keyboard layout and case sensitivity setting."""
        self.keyboard_layout = replacement_config.keyboard_layout
        self.case_sensitive = case_sensitive
    
    def get_random_positions(self, text: str, num_replacements: int) -> List[int]:
        """Get random positions of characters that can be replaced based on keyboard layout."""
        replaceable_positions = [
            i for i, char in enumerate(text)
            if char.lower() in self.keyboard_layout
        ]
        if not replaceable_positions or num_replacements <= 0:
            return []
        return random.sample(replaceable_positions, min(num_replacements, len(replaceable_positions)))
    
    def replace_characters(self, text: str, positions: List[int]) -> str:
        """Replace characters at specified positions with adjacent keyboard characters."""
        result = list(text)
        for pos in positions:
            original_char = result[pos]
            adjacent_chars = self.keyboard_layout[original_char.lower()]
            new_char = random.choice(adjacent_chars)
            
            # Maintain case if case-sensitive
            if self.case_sensitive and original_char.isupper():
                new_char = new_char.upper()
                
            result[pos] = new_char
        return ''.join(result)

class CharReplacement(TextPerturbation):
    """Implements keyboard-based character replacement across entire text."""
    
    def __init__(self, config: PerturbationConfig, replacement_config: ReplacementConfig, case_sensitive: bool = True):
        super().__init__(config)
        self.replacer = CharacterReplacer(replacement_config, case_sensitive)
    
    def perturb(self, text: str) -> str:
        """Replace random characters in the question part of the text with adjacent keyboard characters."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.replacer.get_random_positions(
            question_part,
            self.config.intensity
        )
        
        perturbed_question = self.replacer.replace_characters(question_part, positions)
        return perturbed_question + answer_part
