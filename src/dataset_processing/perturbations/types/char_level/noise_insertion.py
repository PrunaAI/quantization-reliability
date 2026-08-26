from dataclasses import dataclass
from typing import List
import random
import string

from src.dataset_processing.perturbations.base.char_perturbation import CharacterPerturbation
from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

@dataclass
class NoiseConfig:
    """Configuration for noise character insertion."""
    noise_chars: str = string.punctuation + string.digits

class NoiseInserter:
    """Handles noise character insertion operations on a text string."""
    
    def __init__(self, noise_config: NoiseConfig = NoiseConfig()):
        """Initialize with optional set of noise characters."""
        self.noise_chars = noise_config.noise_chars
    
    def get_random_positions(self, text_length: int, num_insertions: int) -> List[tuple[int, int]]:
        """Get random positions and distribute noise insertions among them."""
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
    
    def insert_noise(self, text: str, position_counts: List[tuple[int, int]]) -> str:
        """Insert noise characters at specified positions."""
        result = list(text)
        offset = 0
        for pos, count in sorted(position_counts):  # Sort by position
            noise_chars = [random.choice(self.noise_chars) for _ in range(count)]
            for char in noise_chars:
                result.insert(pos + offset, char)
                offset += 1
        return ''.join(result)

class CharNoiseInsertion(TextPerturbation):
    """Implements random noise character insertion across entire text."""
    
    def __init__(self, config: PerturbationConfig, noise_config: NoiseConfig = NoiseConfig()):
        super().__init__(config)
        self.inserter = NoiseInserter(noise_config)
    
    def perturb(self, text: str) -> str:
        """Insert random noise characters into the question part of the text."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
        
        positions = self.inserter.get_random_positions(
            len(question_part),
            self.config.intensity
        )
        
        perturbed_question = self.inserter.insert_noise(question_part, positions)
        return perturbed_question + answer_part