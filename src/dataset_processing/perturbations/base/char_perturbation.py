import random
from abc import abstractmethod

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation



class CharacterPerturbation(TextPerturbation):
    """Base class for character-level perturbations."""
    @abstractmethod
    def perturb_unit(self, unit: str) -> str:
        """Apply character-level perturbation to a single unit."""
        pass

    def perturb(self, text: str) -> str:
        """Apply character-level perturbation to the entire text."""
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
            
        # Process only question part
        words = question_part.split()
        if not words:
            return text
            
        # Apply character-level perturbation to selected words
        indices = random.sample(range(len(words)), min(self.config.intensity, len(words)))
        for idx in indices:
            words[idx] = self.perturb_unit(words[idx])
        
        # Combine processed question with unchanged answer part
        return " ".join(words) + answer_part