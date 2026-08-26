from pathlib import Path
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
import logging
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.interface import ModelLoaderInterface

logger = logging.getLogger("reliability_eval")

class AWQModelLoader(ModelLoaderInterface):
    """Loader for AWQ quantized models supporting both local and HuggingFace loading."""
    
    def __init__(self):
        # Cache directory for HuggingFace transformers
        self.hf_cache_dir = Path(HF_CACHE_ROOT) / "awq_models"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load an AWQ quantized model based on configuration."""
        logger.info(f"Loading AWQ model with identifier: {config.identifier}")
        
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
            logger.info(f"Loading AWQ model {config.model_path} across multiple devices using max_memory configuration")
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
            logger.info(f"Loading AWQ model {config.model_path} on single device: {config.device}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16,
                device_map=config.device,
                max_memory=max_memory,
                cache_dir=self.hf_cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            # Compile model if CUDA is available (only for single-GPU case)
            if torch.cuda.is_available() and config.apply_compile:
                logger.info(f"Compiling AWQ model with mode: {compile_mode}")
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic
                )
        
        tokenizer = self._load_tokenizer(config)
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
