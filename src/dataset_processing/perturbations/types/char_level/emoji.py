import random

from dataclasses import dataclass
from typing import List, Callable
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

@dataclass
class EmojiConfig:
    """Configuration for emoji operations."""
    emoji_loader: Callable[[], List[str]]
    separator: str = " "

class EmojiHandler:
    """Handles emoji insertion operations on text strings."""
    
    def __init__(self, emoji_config: EmojiConfig, separator: str = " "):
        """Initialize with emoji loading function and separator."""
        self.emoji_loader = emoji_config.emoji_loader
        self.separator = separator
        self.emojis = self.emoji_loader()
    
    def distribute_emojis(self, total_emojis: int, num_positions: int) -> List[int]:
        """Distribute total_emojis randomly across all possible positions."""
        if num_positions <= 0 or total_emojis <= 0:
            return [0] * num_positions
            
        distribution = [0] * num_positions
        
        # Randomly distribute all emojis
        for _ in range(total_emojis):
            position = random.randrange(num_positions)
            distribution[position] += 1
            
        return distribution
    
    def insert_emojis_at_words(self, words: List[str], emoji_counts: List[int]) -> List[str]:
        """Insert specified number of emojis at the end of words based on distribution."""
        result = words.copy()
        for pos, count in enumerate(emoji_counts):
            if count > 0:
                emojis = random.choices(self.emojis, k=count)
                result[pos] = result[pos] + ''.join(emojis)
        return result

class CharEmoji(TextPerturbation):
    """Implements word-based emoji insertion across text."""
    
    def __init__(self, config: PerturbationConfig, emoji_config: EmojiConfig, separator: str = " "):
        super().__init__(config)
        self.handler = EmojiHandler(emoji_config, separator)
    
    def perturb(self, text: str) -> str:
        """Insert emojis at the end of random words in the question part of the text."""
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
            
        words = question_part.split()
        if not words:
            return text
            
        emoji_counts = self.handler.distribute_emojis(
            self.config.intensity,
            len(words)
        )
        
        modified_words = self.handler.insert_emojis_at_words(
            words,
            emoji_counts
        )
        
        perturbed_question = ' '.join(modified_words)
        return perturbed_question + answer_part
