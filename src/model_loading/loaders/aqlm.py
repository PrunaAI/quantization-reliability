from pathlib import Path
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.standard import StandardModelLoader


class AQLMModelLoader(StandardModelLoader):
    """Loader for AQLM quantized models that require float16 precision"""
    
    def __init__(self):
        # Cache directory for HuggingFace transformers
        self.hf_cache_dir = Path(HF_CACHE_ROOT) / "aqlm_models"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Loads an AQLM model with float16 precision and its associated tokenizer"""
        
        # Make a copy of max_memory with integer keys for GPU devices
        max_memory = None
        if config.max_memory is not None:
            max_memory = {}
            for key, value in dict(config.max_memory).items():
                # Convert string device IDs to integers if they're numeric
                if isinstance(key, str) and key.isdigit():
                    max_memory[int(key)] = value
                else:
                    max_memory[key] = value
        
        # Check if we should use multi-GPU loading
        is_multi_gpu = (
            max_memory is not None and
            isinstance(max_memory, dict) and
            len(max_memory) > 1
        )
        
        # Multi-GPU loading path
        if is_multi_gpu:
            print(f"Loading AQLM model {config.model_path} across multiple devices using max_memory configuration")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",  # Use automatic device mapping for multi-GPU
                max_memory=max_memory,
                cache_dir=self.hf_cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            # Skip compilation for multi-GPU models as it's not compatible
        else:
            print(f"Loading AQLM model {config.model_path} on single device: {config.device}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16,
                device_map=config.device,
                max_memory=max_memory,
                cache_dir=self.hf_cache_dir,
                trust_remote_code=config.trust_remote_code
            )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            device_map=config.device,
            cache_dir=self.hf_cache_dir,
            trust_remote_code=config.trust_remote_code
        )

        return model, tokenizer

    def get_model_cache_paths(self, config: ModelConfig):
        return [self.hf_cache_dir / f"models--{config.model_path.replace('/', '--')}"]

    def _load_tokenizer(self, config: ModelConfig) -> AutoTokenizer:
        """Load tokenizer with common configuration."""
        return AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            device_map=config.device,
            cache_dir=self.hf_cache_dir,
            trust_remote_code=config.trust_remote_code
        )