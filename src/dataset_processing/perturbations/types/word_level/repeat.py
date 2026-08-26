from typing import List, Tuple
import random

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.utils.word_processor import WordProcessor

class RepetitionManager:
    """Handler for word repetition operations."""
    def get_repetition_positions(self, length: int, num_repetitions: int) -> List[tuple[int, int]]:
        """Get positions and distribute repetitions among them."""
        if length <= 0 or num_repetitions <= 0:
            return []
            
        # Create distribution of repetitions
        distribution = [0] * length
        
        # Randomly distribute all repetitions
        for _ in range(num_repetitions):
            position = random.randrange(length)
            distribution[position] += 1
        
        # Return only positions that got at least one repetition
        return [(pos, count) for pos, count in enumerate(distribution) if count > 0]

    def insert_repetition(self, words: List[str], position: int, count: int) -> None:
        """Insert word repetitions at given position."""
        word = words[position]
        # Insert the word 'count' times after its original occurrence
        for _ in range(count):
            words.insert(position + 1, word)

class WordRepetition(TextPerturbation):
    """Implementation of keyword repetition."""
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.word_processor = WordProcessor()
        self.repetition_manager = RepetitionManager()

    def _process_words(self, words: List[str]) -> List[Tuple[str, object, object]]:
        """Process words to handle punctuation."""
        return [self.word_processor.extract_punctuation(word) for word in words]

    def _perform_repetitions(
        self,
        clean_words: List[str],
        processed_words: List[Tuple[str, object, object]]
    ) -> None:
        """Perform word repetition operations."""
        position_counts = self.repetition_manager.get_repetition_positions(
            len(clean_words),
            self.config.intensity
        )
        # Sort positions in reverse order to avoid affecting subsequent insertions
        for position, count in sorted(position_counts, reverse=True):
            self.repetition_manager.insert_repetition(clean_words, position, count)
            self.repetition_manager.insert_repetition(processed_words, position, count)

    def perturb(self, text: str) -> str:
        """Apply word repetition perturbation to text."""
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text

        # Process only question part
        words = question_part.split()
        if not words:
            return text
            
        # Process words and maintain punctuation
        processed_words = self._process_words(words)
        clean_words = [word for word, _, _ in processed_words]
        
        # Perform repetition operations
        self._perform_repetitions(clean_words, processed_words)
        
        # Restore punctuation
        result_words = [
            self.word_processor.restore_punctuation(word, start_punct, end_punct)
            for word, start_punct, end_punct in processed_words
        ]
        
        # Combine processed question with unchanged answer part
        return " ".join(result_words) + answer_part