from abc import ABC, abstractmethod
from typing import Tuple

from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig


class TextPerturbation(ABC):
    """Abstract base class for text perturbation operations."""
    def __init__(self, config: PerturbationConfig):
        self.config = config

    def split_question_answer(self, text: str) -> Tuple[str, str]:
        """Split text into question and answer parts at 'A.'."""
        marker = '\nA. '
        if marker in text:
            parts = text.split(marker, 1)
            return parts[0], marker + parts[1]
        return text, ""

    @abstractmethod
    def perturb(self, text: str) -> str:
        """Apply perturbation to the input text."""
        pass