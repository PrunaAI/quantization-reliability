from dataclasses import dataclass
from typing import Dict, Optional
from src.model_loading.common.identifier import ModelIdentifier


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    identifier: ModelIdentifier
    device: str = "cuda"
    max_memory: Optional[Dict[str, str]] = None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True
    apply_compile: bool = True