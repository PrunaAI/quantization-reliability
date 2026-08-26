from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.model_loading.common.model_config import ModelConfig


class ModelLoaderInterface(ABC):
    """Interface for model loaders following Strategy pattern"""
    @abstractmethod
    def load_model(self, config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer according to configuration"""
        pass

    def get_model_cache_paths(self, config: ModelConfig) -> List[Path]:
        """Return all cache paths for this model that should be cleaned up.
        Returns an empty list for loaders where cache should not be deleted (e.g. GPTQ)."""
        return []
