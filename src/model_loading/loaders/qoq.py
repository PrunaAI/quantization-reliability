from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.standard import StandardModelLoader

class QoQModelLoader(StandardModelLoader):
    """Specialized loader for QoQ-quantized models."""
    def load_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Loads a QoQ-quantized model with appropriate configuration settings."""
        
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
            print(f"Loading QoQ model {config.model_path} across multiple devices using max_memory configuration")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype="auto",
                device_map="auto",  # Use automatic device mapping for multi-GPU
                max_memory=max_memory,
                cache_dir=self.hf_cache_dir,
                trust_remote_code=config.trust_remote_code,
                use_safetensors=False,
                revision="main"
            )
            # Skip compilation for multi-GPU models as it's not compatible
        else:
            print(f"Loading QoQ model {config.model_path} on single device: {config.device}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype="auto",
                device_map=config.device,
                max_memory=max_memory,
                cache_dir=self.hf_cache_dir,
                trust_remote_code=config.trust_remote_code,
                use_safetensors=False,
                revision="main"
            )
            
            # Compile model if CUDA is available (only for single-GPU case)
            if torch.cuda.is_available() and config.apply_compile:
                print(f"Compiling QoQ model with mode: {compile_mode}")
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic
                )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            device_map=config.device,
            cache_dir=self.hf_cache_dir,
            trust_remote_code=config.trust_remote_code
        )
        
        return model, tokenizer
