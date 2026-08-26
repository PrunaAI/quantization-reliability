from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPaths:
    """Dataclass containing model and tokenizer paths"""
    model_path: str
    tokenizer_path: Optional[str] = None
