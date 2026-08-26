from dataclasses import dataclass
from typing import List

from src.dataset_processing.perplexity.common.config.base_configs import PerplexityDatasetConfig
from src.dataset_processing.perplexity.common.models.dataset_entry import PerplexityDatasetEntry


@dataclass
class PerplexityDatasetResult:
    """Result container for processed perplexity dataset."""
    entries: List[PerplexityDatasetEntry]
    config: PerplexityDatasetConfig