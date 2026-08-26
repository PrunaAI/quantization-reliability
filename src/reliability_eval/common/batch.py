from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch


@dataclass
class BatchData:
    """Contains processed batch data for model evaluation."""
    repeated_queries: np.ndarray
    repeated_answers: np.ndarray
    model_inputs: Dict[str, torch.Tensor]
