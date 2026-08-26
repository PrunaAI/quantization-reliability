import random

from dataclasses import dataclass
from typing import Dict, List
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

@dataclass
class SubstitutionConfig:
    """Configuration for character substitution operations."""
    char_map: Dict[str, str]
    case_sensitive: bool = True

class CharacterSubstitutor:
    """Handles character substitution operations on a text string."""
    
    def __init__(self, substitution_config: SubstitutionConfig, case_sensitive: bool = True):
        """Initialize with character mapping and case sensitivity setting."""
        self.char_map = substitution_config.char_map
        self.case_sensitive = case_sensitive
    
    def get_random_positions(self, text: str, num_substitutions: int) -> List[int]:
        """Get random positions of characters that can be substituted based on mapping."""
        substitutable_positions = [
            i for i, char in enumerate(text)
            if char.upper() in self.char_map
        ]
        if not substitutable_positions or num_substitutions <= 0:
            return []
        return random.sample(substitutable_positions, min(num_substitutions, len(substitutable_positions)))
    
    def substitute_characters(self, text: str, positions: List[int]) -> str:
        """Substitute characters at specified positions using the character map."""
        result = list(text)
        for pos in positions:
            original_char = result[pos]
            new_char = self.char_map[original_char.upper()]
            
            # Maintain case if case-sensitive
            if self.case_sensitive and original_char.islower():
                new_char = new_char.lower()
                
            result[pos] = new_char
        return ''.join(result)

class CharSubstitution(TextPerturbation):
    """Implements mapped character substitution across entire text."""
    
    def __init__(self, config: PerturbationConfig, substitution_config: SubstitutionConfig, case_sensitive: bool = True):
        super().__init__(config)
        self.substitutor = CharacterSubstitutor(substitution_config, case_sensitive)
    
    def perturb(self, text: str) -> str:
        """Substitute random characters in the question part of the text using the character map."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.substitutor.get_random_positions(
            question_part,
            self.config.intensity
        )
        
        perturbed_question = self.substitutor.substitute_characters(question_part, positions)
        return perturbed_question + answer_part
