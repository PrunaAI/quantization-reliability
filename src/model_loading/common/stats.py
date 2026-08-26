from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelStats:
    """Statistics about loaded model"""
    cache_dir: Optional[str]
    model_max_length: int
    dtype: Any
    device: str
    num_parameters: int
    memory_footprint: float
    vocab_size: int
    pad_token_id: int
    special_tokens: Dict
