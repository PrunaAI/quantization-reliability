from dataclasses import dataclass
from typing import Optional
from src.dataset_processing.perplexity.common.enums.dataset_types import PerplexityDatasetType


@dataclass
class PerplexityDatasetConfig:
    """Base configuration for perplexity datasets."""
    dataset_type: PerplexityDatasetType
    split: str
    n_samples: Optional[int]
    seq_length: int
    batch_size: int
    seed: int
    tokenizer_name: str