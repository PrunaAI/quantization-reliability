from typing import List
import random
import string
from dataclasses import dataclass

from src.dataset_processing.perturbations.base.char_perturbation import CharacterPerturbation
from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

@dataclass
class CharInsertionConfig:
    """Configuration for character insertion operations."""
    char_set: str = string.ascii_letters

class CharacterInserter:
    """Handles character insertion operations on a text string."""
    
    def __init__(self, char_set: str = string.ascii_letters):
        """Initialize with optional character set for insertions."""
        self.char_set = char_set
    
    def get_random_positions(self, text_length: int, num_insertions: int) -> List[int]:
        """Get random positions and distribute insertions among them."""
        if text_length < 0 or num_insertions <= 0:
            return []
            
        # Create distribution of insertions
        possible_positions = range(text_length + 1)
        distribution = [0] * (text_length + 1)
        
        # Randomly distribute all insertions
        for _ in range(num_insertions):
            position = random.choice(possible_positions)
            distribution[position] += 1
        
        # Return only positions that got at least one insertion
        return [(pos, count) for pos, count in enumerate(distribution) if count > 0]
    
    def insert_characters(self, text: str, position_counts: List[tuple[int, int]]) -> str:
        """Insert random characters at specified positions."""
        result = list(text)
        offset = 0
        for pos, count in sorted(position_counts):  # Sort by position
            chars = [random.choice(self.char_set) for _ in range(count)]
            for char in chars:
                result.insert(pos + offset, char)
                offset += 1
        return ''.join(result)

class CharInsertion(TextPerturbation):
    """Implements random character insertion across entire text."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.inserter = CharacterInserter()
    
    def perturb(self, text: str) -> str:
        """Insert random characters into the question part of the text."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.inserter.get_random_positions(
            len(question_part),
            self.config.intensity
        )
        
        perturbed_question = self.inserter.insert_characters(question_part, positions)
        return perturbed_question + answer_part
