from dataclasses import dataclass
from typing import Optional

from src.reliability_eval.common.generation import GenerationStrategy
from src.reliability_eval.prompting.types import PromptStrategy


@dataclass
class GenerationExperimentConfig:
    """
    Dataclass to store generation configuration for language models.
    """
    # Required parameters
    prompt_strategy: PromptStrategy = PromptStrategy.DIRECT_COMPLETION
    num_repeats: int = 5
    max_new_tokens: int = 25
    temperature: float = 0.1
    generation_strategy: GenerationStrategy = GenerationStrategy.GREEDY_SEARCH
    top_k_concentration: int = 3
    
    # Optional parameters
    dataset_name: Optional[str] = None
    num_shots: Optional[int] = 0
    num_beams: Optional[int] = None
    num_return_sequences: Optional[int] = None
    num_beam_groups: Optional[int] = None
    penalty_alpha: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None