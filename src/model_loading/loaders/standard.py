from pathlib import Path
from typing import Optional, Tuple, Union
from src.model_loading.common.model_enums import ModelFamily
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.interface import ModelLoaderInterface

# Conditional import for VLM models
try:
    from transformers import Qwen3VLForConditionalGeneration
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None


class StandardModelLoader(ModelLoaderInterface):
    """Loader for standard (non-quantized) models and Vision-Language Models"""
    
    def __init__(self):
        # Cache directories for HuggingFace transformers
        self.hf_cache_dir = Path(HF_CACHE_ROOT) / "standard_models"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.vlm_cache_dir = Path(HF_CACHE_ROOT) / "vlm_models"
        self.vlm_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_vlm_model(self, config: ModelConfig) -> bool:
        """Check if the model is a Vision-Language Model"""
        return config.identifier.family in [ModelFamily.QWEN3VL, ModelFamily.QWEN25VL]

    def get_model_cache_paths(self, config: ModelConfig):
        cache_dir = self.vlm_cache_dir if self._is_vlm_model(config) else self.hf_cache_dir
        return [cache_dir / f"models--{config.model_path.replace('/', '--')}"]
    
    def load_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False
    ) -> Union[
        Tuple[AutoModelForCausalLM, AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]:
        
        # Check if this is a VLM model
        is_vlm = self._is_vlm_model(config)
        
        if is_vlm:
            if not VLM_AVAILABLE:
                raise ImportError(
                    f"Qwen3VLForConditionalGeneration is not available in your transformers version. "
                    f"Please upgrade transformers to use VLM models: pip install --upgrade transformers"
                )
            return self._load_vlm_model(config, compile_mode, fullgraph, dynamic)
        else:
            return self._load_standard_model(config, compile_mode, fullgraph, dynamic)
    
    def _load_vlm_model(
        self,
        config: ModelConfig,
        compile_mode: str,
        fullgraph: bool,
        dynamic: bool
    ) -> Tuple[object, AutoProcessor]:
        """Load Vision-Language Model"""
        
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
            print(f"Loading VLM model {config.model_path} across multiple devices using max_memory configuration")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_path,
                torch_dtype="auto",
                device_map="auto",
                max_memory=max_memory,
                cache_dir=self.vlm_cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            # Skip compilation for multi-GPU models as it's not compatible
        
        # Original single-GPU loading path
        else:
            print(f"Loading VLM model {config.model_path} on single device: {config.device}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_path,
                torch_dtype="auto",
                device_map=config.device,
                max_memory=max_memory,
                cache_dir=self.vlm_cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            # Compile model if CUDA is available (only for single-GPU case)
            if torch.cuda.is_available() and config.apply_compile:
                print(f"Compiling VLM model with mode: {compile_mode}")
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic
                )
        
        # Load processor (instead of tokenizer for VLM)
        processor = AutoProcessor.from_pretrained(
            config.tokenizer_path or config.model_path,
            cache_dir=self.vlm_cache_dir,
            trust_remote_code=config.trust_remote_code
        )
        
        return model, processor
    
    def _load_standard_model(
        self,
        config: ModelConfig,
        compile_mode: str,
        fullgraph: bool,
        dynamic: bool
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load standard (non-VLM) model"""
        
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
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                cache_dir=self.hf_cache_dir,
                torch_dtype="auto",
                device_map="auto",  # Use automatic device mapping for multi-GPU
                max_memory=max_memory,
                trust_remote_code=config.trust_remote_code
            )
            # Skip compilation for multi-GPU models as it's not compatible
        
        # Original single-GPU loading path
        else:
            print(f"Loading model {config.model_path} on single device: {config.device}")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                cache_dir=self.hf_cache_dir,
                torch_dtype="auto",
                device_map=config.device,
                max_memory=max_memory,
                trust_remote_code=config.trust_remote_code
            )
            
            # Compile model if CUDA is available (only for single-GPU case)
            if torch.cuda.is_available() and config.apply_compile:
                print(f"Compiling model with mode: {compile_mode}")
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic
                )
        
        # Load tokenizer (same for both paths)
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            device_map=config.device,
            cache_dir=self.hf_cache_dir,
            trust_remote_code=config.trust_remote_code
        )
        
        return model, tokenizer
