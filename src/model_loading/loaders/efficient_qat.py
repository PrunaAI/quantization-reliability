from pathlib import Path
from typing import Tuple
from transformers import AutoTokenizer
from gptqmodel import GPTQModel
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.interface import ModelLoaderInterface


class EfficientQATModelLoader(ModelLoaderInterface):
    """Loader for EfficientQAT quantized models (GPTQ/BitBLAS formats)"""
    
    def __init__(self):
        # Cache directory for HuggingFace transformers
        self.hf_cache_dir = Path(HF_CACHE_ROOT) / "efficientqat"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False
    ) -> Tuple[GPTQModel, AutoTokenizer]:
        
        print(f"Loading EfficientQAT model {config.model_path}")
        
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
            print(f"Loading model {config.model_path} across multiple devices using max_memory configuration")
            model = GPTQModel.from_quantized(
                config.model_path,
                cache_dir=self.hf_cache_dir,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=config.trust_remote_code
            )
            # Skip compilation for multi-GPU models as it's not compatible
        
        # Original single-GPU loading path
        else:
            print(f"Loading model {config.model_path} on single device: {config.device}")
            model = GPTQModel.from_quantized(
                config.model_path,
                cache_dir=self.hf_cache_dir,
                device=config.device if config.device != "auto" else "cuda:0",
                trust_remote_code=config.trust_remote_code
            )
            
            # Note: torch.compile typically doesn't work well with quantized models
            # Skip compilation for EfficientQAT models
            if config.apply_compile:
                print("Warning: Skipping torch.compile for quantized model (not recommended)")
        
        # Load tokenizer (same for both paths)
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            use_fast=True,
            cache_dir=self.hf_cache_dir,
            trust_remote_code=config.trust_remote_code
        )
        
        return model, tokenizer
