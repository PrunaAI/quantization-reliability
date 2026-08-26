from typing import List, Set
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

class InternetSlangManager:
    """Handler for internet slang operations."""
    SLANG_TERMS: Set[str] = {
        'lol', 'rofl', 'idk', 'tbh', 'imo', 'fyi', 'brb', 'afk', 'tl;dr',
        'wtf', 'omg', 'smh', 'yolo', 'fomo', 'irl', 'tfw', 'ftw', 'imho',
        'iirc', 'asap', 'thx', 'hmu', 'xoxo', 'fwiw', 'ftfy', 'ama', 'eli5'
    }

    def get_random_slang(self) -> str:
        """Get random internet slang term."""
        return random.choice(list(self.SLANG_TERMS))

    def get_insertion_positions(self, length: int, num_insertions: int) -> List[tuple[int, int]]:
        """Get positions and distribute slang insertions among them."""
        if length < 0 or num_insertions <= 0:
            return []
            
        # Create distribution of insertions
        possible_positions = range(length + 1)
        distribution = [0] * (length + 1)
        
        # Randomly distribute all insertions
        for _ in range(num_insertions):
            position = random.choice(possible_positions)
            distribution[position] += 1
        
        # Return only positions that got at least one insertion
        return [(pos, count) for pos, count in enumerate(distribution) if count > 0]

class InternetSlangInsertion(TextPerturbation):
    """Implementation of internet slang insertion."""
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.slang_manager = InternetSlangManager()

    def _perform_insertions(self, words: List[str]) -> List[str]:
        """Perform slang insertion operations."""
        position_counts = self.slang_manager.get_insertion_positions(
            len(words),
            self.config.intensity
        )
        result = words.copy()
        offset = 0
        for pos, count in sorted(position_counts):  # Sort by position
            for _ in range(count):
                slang_term = self.slang_manager.get_random_slang()
                result.insert(pos + offset, slang_term)
                offset += 1
        return result

    def perturb(self, text: str) -> str:
        """Apply internet slang insertion to text."""
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text

        # Process only question part
        words = question_part.split()
        modified_words = self._perform_insertions(words)
        
        # Combine processed question with unchanged answer part
        return " ".join(modified_words) + answer_part
