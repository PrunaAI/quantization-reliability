from dataclasses import dataclass
from typing import List, Optional

from src.dataset_processing.perturbations.enums import PerturbationType



@dataclass
class PerturbationConfig:
    """Configuration for perturbation operations."""
    type: PerturbationType
    intensity: int
    taxonomies: Optional[List[str]] = None