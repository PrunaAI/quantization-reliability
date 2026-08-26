import numpy as np
from dataclasses import dataclass
from typing import Iterator, List

import logging
from src.reliability_eval.pipeline.config import BatchConfig


logger = logging.getLogger("reliability_eval")

@dataclass
class Batch:
    """Container for batch data."""
    queries: List[str]
    answers: np.ndarray
    batch_idx: List[int]

class DataLoader:
    """Handles batched data loading and iteration."""

    def __init__(self, queries: List[str], answers: np.ndarray, batch_config: BatchConfig):
        """Initialize data loader with queries and answers."""
        self.queries = queries
        self.answers = answers
        self.batch_config = batch_config
        self.indices = np.arange(len(queries))
        
        if self.batch_config.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        start = 0
        total = len(self.queries)

        while start < total:
            end = min(start + self.batch_config.batch_size, total)
            if end - start < self.batch_config.batch_size and self.batch_config.drop_last:
                break

            batch_indices = self.indices[start:end]
            yield Batch(
                queries=[self.queries[i] for i in batch_indices],
                answers=self.answers[batch_indices],
                batch_idx=batch_indices.tolist()
            )
            start = end
