from typing import Dict, Tuple, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.model_loading.common.identifier import ModelIdentifier
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.manager import ModelManager

def load_model_for_evaluation(
    model_identifier: ModelIdentifier,
    device: str = "cuda",
    max_memory: Optional[Dict[str, str]] = None,
    apply_compile: bool = True,
    model_save_path: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load and prepare a model and tokenizer for reliability evaluation."""
    config = ModelConfig(
        identifier=model_identifier,
        device=device,
        max_memory=max_memory,
        apply_compile=apply_compile,
    )
    manager = ModelManager(model_save_path=model_save_path)
    model, tokenizer = manager.load_model(config)

    model.eval()
    
    return model, tokenizer
