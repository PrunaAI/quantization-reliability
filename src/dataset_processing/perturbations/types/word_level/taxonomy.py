from typing import List

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig


class TaxonomyManager:
    """Handler for taxonomy operations."""
    
    def __init__(self, taxonomies: List[str]):
        self.taxonomy_list = taxonomies
    
    @property
    def correct_taxonomy(self) -> str:
        """Get the correct taxonomy (last item)."""
        return self.taxonomy_list[-1]
    
    @property
    def incorrect_taxonomies(self) -> List[str]:
        """Get incorrect taxonomies (all but last item)."""
        return self.taxonomy_list[:-1]
    
    def get_taxonomies_for_insertion(self, num_insertions: int, positive: bool) -> List[str]:
        """Get taxonomies to insert based on type."""
        if positive:
            return [self.correct_taxonomy] * num_insertions
        else:
            return self.incorrect_taxonomies[:num_insertions]

class BaseTaxonomyPerturbation(TextPerturbation):
    """Base class for taxonomy perturbations."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        if not config.taxonomies:
            raise ValueError("Taxonomies must be provided in PerturbationConfig for taxonomy perturbations")
        self.taxonomy_manager = TaxonomyManager(config.taxonomies)
    
    def _prefix_taxonomies(self, text: str, taxonomies: List[str]) -> str:
        """Add taxonomy prefixes to text."""
        result = text
        for taxonomy in taxonomies:
            result = f"{taxonomy}. {result}"
        return result

class PositiveTaxonomyPerturbation(BaseTaxonomyPerturbation):
    """Implementation of positive taxonomy perturbation."""
    
    def perturb(self, text: str) -> str:
        """Apply positive taxonomy perturbation to text."""
        taxonomies = self.taxonomy_manager.get_taxonomies_for_insertion(
            self.config.intensity,
            positive=True
        )
        return self._prefix_taxonomies(text, taxonomies)

class NegativeTaxonomyPerturbation(BaseTaxonomyPerturbation):
    """Implementation of negative taxonomy perturbation."""
    
    def perturb(self, text: str) -> str:
        """Apply negative taxonomy perturbation to text."""
        taxonomies = self.taxonomy_manager.get_taxonomies_for_insertion(
            self.config.intensity,
            positive=False
        )
        return self._prefix_taxonomies(text, taxonomies)