from dataclasses import dataclass, field
from typing import Dict, List

from src.dataset_processing.common.constants import ALL_PERTURBATION_INTENSITIES
from src.dataset_processing.perturbations.enums import PerturbationType


@dataclass
class MergeConfig:
    """Configuration for dataset merging operations"""
    base_perturbation_intensities: Dict[PerturbationType, List[int]] = field(default_factory=lambda: ALL_PERTURBATION_INTENSITIES)
    num_perturbation_types: int = 1
    
    def __post_init__(self):
        """Calculate number of perturbation types as the sum of all perturbation intensities"""
        self.num_perturbation_types = sum(len(intensities) for intensities in self.base_perturbation_intensities.values())
