from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig
)
import logging
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.interface import ModelLoaderInterface
from src.model_loading.common.model_enums import BitPrecision, ModelFamily

logger = logging.getLogger("reliability_eval")

# Conditional import for VLM models
try:
    from transformers import Qwen3VLForConditionalGeneration
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None

class BitsAndBytesModelLoader(ModelLoaderInterface):
    """Loader for BitsAndBytes quantized models supporting both local and HuggingFace loading, including VLM models."""
    
    def __init__(self):
        # Cache directory for BitsAndBytes quantized outputs
        self.bnb_cache_dir = Path(HF_CACHE_ROOT) / "bnb_models"
        self.bnb_cache_dir.mkdir(parents=True, exist_ok=True)

        self.vlm_cache_dir = Path(HF_CACHE_ROOT) / "bnb_vlm_models"
        self.vlm_cache_dir.mkdir(parents=True, exist_ok=True)

        # Shared base model cache — reused across all quantization methods
        self.standard_cache_dir = Path(HF_CACHE_ROOT) / "standard_models"
        self.standard_cache_dir.mkdir(parents=True, exist_ok=True)

        self.standard_vlm_cache_dir = Path(HF_CACHE_ROOT) / "vlm_models"
        self.standard_vlm_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_vlm_model(self, config: ModelConfig) -> bool:
        """Check if the model is a Vision-Language Model"""
        return config.identifier.family in [ModelFamily.QWEN3VL, ModelFamily.QWEN25VL]
    
    def get_model_cache_paths(self, config: ModelConfig):
        is_vlm = self._is_vlm_model(config)
        base_cache_dir = self.standard_vlm_cache_dir if is_vlm else self.standard_cache_dir
        hf_base_path = base_cache_dir / f"models--{config.model_path.replace('/', '--')}"
        quantized_path = self._get_cache_path(config.model_path, config.identifier.bits.value, is_vlm)
        return [hf_base_path, quantized_path]

    def _get_cache_path(self, model_path: str, bits: int, is_vlm: bool = False) -> Path:
        """Generate unique cache path for BnB model based on name and bits."""
        model_name = model_path.split('/')[-1]
        cache_dir = self.vlm_cache_dir if is_vlm else self.bnb_cache_dir
        suffix = "vlm" if is_vlm else "bnb"
        return cache_dir / f"{model_name}_{bits}bit_{suffix}"
    
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
        """Load a model based on configuration using either local or HF loading strategy."""
        logger.info(f"Loading BitsAndBytes model with identifier: {config.identifier}")
        
        # Check if this is a VLM model
        is_vlm = self._is_vlm_model(config)
        
        if is_vlm:
            if not VLM_AVAILABLE:
                raise ImportError(
                    f"Qwen3VLForConditionalGeneration is not available in your transformers version. "
                    f"Please upgrade transformers to use VLM models: pip install --upgrade transformers"
                )
            logger.warning("VLM models with BitsAndBytes quantization - note that quantization may not be fully supported for VLM architectures")
        
        if config.identifier.is_local:
            return self._load_local_model(
                config=config,
                compile_mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                is_vlm=is_vlm
            )

        # For HF models, use cache-based approach like GPTQ
        if not config.identifier.bits or config.identifier.bits not in [BitPrecision.INT4, BitPrecision.INT8]:
            raise ValueError(f"BitsAndBytes requires 4 or 8 bit precision, got: {config.identifier.bits}")

        cache_path = self._get_cache_path(config.model_path, config.identifier.bits.value, is_vlm)
        logger.info(f"Loading {config.identifier.bits.value}-bit BnB {'VLM' if is_vlm else ''} model with cache path: {cache_path}")

        # Try loading from cache first
        cached_model = self._load_bnb_from_cache(cache_path, config, is_vlm)
        if cached_model is not None:
            model, tokenizer_or_processor = cached_model
            
            # Apply compilation if needed
            if torch.cuda.is_available() and config.apply_compile:
                # Check for multi-GPU to skip compilation
                max_memory = None
                if config.max_memory is not None:
                    max_memory = {}
                    for key, value in dict(config.max_memory).items():
                        if isinstance(key, str) and key.isdigit():
                            max_memory[int(key)] = value
                        else:
                            max_memory[key] = value
                
                is_multi_gpu = (
                    max_memory is not None and
                    isinstance(max_memory, dict) and
                    len(max_memory) > 1
                )
                
                if not is_multi_gpu:
                    model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph, dynamic=dynamic)
            
            return model, tokenizer_or_processor

        # If not in cache, load and save to cache
        return self._load_and_save_hf_model(
            config=config,
            cache_path=cache_path,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            is_vlm=is_vlm
        )
    
    def _load_local_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False,
        is_vlm: bool = False
    ) -> Union[
        Tuple[AutoModelForCausalLM, AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]:
        """Load a local model using standard loading strategy."""
        cache_dir = self.vlm_cache_dir if is_vlm else self.bnb_cache_dir
        
        if is_vlm:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16,
                device_map=config.device,
                max_memory=config.max_memory,
                cache_dir=cache_dir,
                trust_remote_code=config.trust_remote_code
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=torch.float16,
                device_map=config.device,
                max_memory=config.max_memory,
                cache_dir=cache_dir,
                trust_remote_code=config.trust_remote_code
            )
        
        # Compile model if CUDA is available
        if torch.cuda.is_available() and config.apply_compile:
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
        
        # Load tokenizer or processor
        if is_vlm:
            processor = self._load_processor(config)
            return model, processor
        else:
            tokenizer = self._load_tokenizer(config)
            return model, tokenizer
    
    def _load_and_save_hf_model(
        self,
        config: ModelConfig,
        cache_path: Path,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False,
        is_vlm: bool = False
    ) -> Union[
        Tuple[AutoModelForCausalLM, AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]:
        """Load a HuggingFace model using BitsAndBytes quantization."""
        bits = config.identifier.bits
        if not bits or bits not in [BitPrecision.INT4, BitPrecision.INT8]:
            raise ValueError(f"BitsAndBytes requires 4 or 8 bit precision, got: {bits}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(bits == BitPrecision.INT4),
            load_in_8bit=(bits == BitPrecision.INT8)
        )
        
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
        
        cache_dir = self.standard_vlm_cache_dir if is_vlm else self.standard_cache_dir

        # Multi-GPU loading path
        if is_multi_gpu:
            logger.info(f"Loading HF BitsAndBytes {'VLM' if is_vlm else ''} model {config.model_path} across multiple devices using max_memory configuration")
            if is_vlm:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    config.model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory=max_memory,
                    cache_dir=cache_dir,
                    trust_remote_code=config.trust_remote_code
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory=max_memory,
                    cache_dir=cache_dir,
                    trust_remote_code=config.trust_remote_code
                )
            # Skip compilation for multi-GPU models as it's not compatible
        else:
            logger.info(f"Loading HF BitsAndBytes {'VLM' if is_vlm else ''} model {config.model_path} on single device: {config.device}")
            if is_vlm:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    config.model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map=config.device,
                    max_memory=max_memory,
                    cache_dir=cache_dir,
                    trust_remote_code=config.trust_remote_code
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map=config.device,
                    max_memory=max_memory,
                    cache_dir=cache_dir,
                    trust_remote_code=config.trust_remote_code
                )
            
            # Compile model if CUDA is available (only for single-GPU case)
            if torch.cuda.is_available() and config.apply_compile:
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic
                )
        
        # Load tokenizer or processor
        if is_vlm:
            processor = self._load_processor(config)
            tokenizer_or_processor = processor
        else:
            tokenizer = self._load_tokenizer(config)
            tokenizer_or_processor = tokenizer
        
        try:
            logger.info(f"Saving BnB model to cache: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(cache_path)
            tokenizer_or_processor.save_pretrained(cache_path)
            logger.info(f"Successfully saved BnB model to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save BnB model to cache: {str(e)}")
            logger.warning("Continuing without caching...")

        return model, tokenizer_or_processor
    
    def _load_tokenizer(self, config: ModelConfig) -> AutoTokenizer:
        """Load tokenizer with common configuration."""
        return AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            device_map=config.device,
            cache_dir=self.standard_cache_dir,
            trust_remote_code=config.trust_remote_code
        )

    def _load_processor(self, config: ModelConfig) -> AutoProcessor:
        """Load processor for VLM models."""
        return AutoProcessor.from_pretrained(
            config.tokenizer_path or config.model_path,
            cache_dir=self.standard_vlm_cache_dir,
            trust_remote_code=config.trust_remote_code
        )
    
    def _load_bnb_from_cache(
        self,
        cache_path: Path,
        config: ModelConfig,
        is_vlm: bool = False
    ) -> Optional[Union[
        Tuple[AutoModelForCausalLM, AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]]:
        """Try loading BnB model from cache path."""
        if not cache_path.exists():
            return None
        try:
            logger.info(f"Loading BnB {'VLM' if is_vlm else ''} model from cache: {cache_path}")
            
            # Make a copy of max_memory with integer keys for GPU devices
            max_memory = None
            if config.max_memory is not None:
                max_memory = {}
                for key, value in dict(config.max_memory).items():
                    if isinstance(key, str) and key.isdigit():
                        max_memory[int(key)] = value
                    else:
                        max_memory[key] = value
            
            is_multi_gpu = (
                max_memory is not None and
                isinstance(max_memory, dict) and
                len(max_memory) > 1
            )
            
            device_map = "auto" if is_multi_gpu else config.device
            cache_dir = self.vlm_cache_dir if is_vlm else self.bnb_cache_dir
            
            if is_vlm:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    cache_path,
                    device_map=device_map,
                    max_memory=max_memory,
                    trust_remote_code=config.trust_remote_code,
                    torch_dtype=torch.float16
                )
                
                processor = AutoProcessor.from_pretrained(
                    cache_path,
                    cache_dir=cache_dir,
                    trust_remote_code=config.trust_remote_code
                )
                
                return model, processor
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cache_path,
                    device_map=device_map,
                    max_memory=max_memory,
                    trust_remote_code=config.trust_remote_code,
                    torch_dtype=torch.float16
                )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    cache_path,
                    trust_remote_code=config.trust_remote_code
                )
                
                return model, tokenizer
            
        except Exception as e:
            logger.warning(f"Failed to load BnB model from cache: {e}")
            return None
