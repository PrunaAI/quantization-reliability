import json
import os
from typing import Tuple, Optional, Union
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoProcessor,
    AutoConfig,
    QuantoConfig
)
from safetensors.torch import load_file
from accelerate import init_empty_weights
import logging
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.common.model_enums import ModelFamily
from src.model_loading.loaders.interface import ModelLoaderInterface

logger = logging.getLogger("reliability_eval")

# Conditional import for VLM models
try:
    from transformers import Qwen3VLForConditionalGeneration
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None

class QuantoModelLoader(ModelLoaderInterface):
    """Loader for QUANTO quantized models supporting both local and HuggingFace loading, including VLM models."""
    
    def __init__(self):
        # Cache directory for Quanto quantized outputs
        self.quanto_cache_dir = Path(HF_CACHE_ROOT) / "quanto_models"
        self.quanto_cache_dir.mkdir(parents=True, exist_ok=True)

        self.vlm_cache_dir = Path(HF_CACHE_ROOT) / "quanto_vlm_models"
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
        bits = config.identifier.bits.value if config.identifier.bits else 8
        quantized_path = self._get_cache_path(config.model_path, bits, is_vlm)
        return [hf_base_path, quantized_path]

    def _get_cache_path(self, model_path: str, bits: int, is_vlm: bool = False) -> Path:
        """Generate unique cache path for Quanto model based on name and bits."""
        model_name = model_path.split('/')[-1]
        cache_dir = self.vlm_cache_dir if is_vlm else self.quanto_cache_dir
        suffix = "vlm" if is_vlm else "quanto"
        return cache_dir / f"{model_name}_{bits}bit_{suffix}"
        
    def _load_quanto_from_cache(
        self,
        cache_path: Path,
        config: ModelConfig,
        is_vlm: bool = False
    ) -> Optional[Union[
        Tuple[AutoModelForCausalLM, AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]]:
        """Try loading Quanto model from cache path."""
        if not cache_path.exists():
            return None
        
        # Check for required files
        safetensors_path = cache_path / "model.safetensors"
        quantmap_path = cache_path / "quantization_map.json"
        
        if not safetensors_path.exists() or not quantmap_path.exists():
            logger.warning(f"Cache incomplete at {cache_path}, missing required files")
            return None
            
        try:
            logger.info(f"Loading Quanto {'VLM' if is_vlm else ''} model from cache: {cache_path}")
            
            # Load using the same logic as _load_local_model
            state_dict = load_file(str(safetensors_path))
            with open(quantmap_path, 'r') as f:
                quantization_map = json.load(f)
            
            cache_dir = self.standard_vlm_cache_dir if is_vlm else self.standard_cache_dir

            # Initialize model
            model_config = AutoConfig.from_pretrained(
                config.tokenizer_path or config.model_path,
                trust_remote_code=config.trust_remote_code,
                cache_dir=cache_dir
            )
            
            with init_empty_weights():
                if is_vlm:
                    model = Qwen3VLForConditionalGeneration.from_config(
                        model_config,
                        trust_remote_code=config.trust_remote_code
                    )
                else:
                    model = AutoModelForCausalLM.from_config(
                        model_config, 
                        trust_remote_code=config.trust_remote_code
                    )
            
            # Apply quantization
            from optimum.quanto import requantize
            requantize(model, state_dict, quantization_map, config.device)
            
            if is_vlm:
                processor = AutoProcessor.from_pretrained(
                    cache_path,
                    cache_dir=cache_dir,
                    trust_remote_code=config.trust_remote_code
                )
                return model, processor
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    cache_path,
                    trust_remote_code=config.trust_remote_code
                )
                return model, tokenizer
            
        except Exception as e:
            logger.warning(f"Failed to load Quanto model from cache: {e}")
            return None
    
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
        logger.info(f"Loading QUANTO model with identifier: {config.identifier}")
        
        # Check if this is a VLM model
        is_vlm = self._is_vlm_model(config)
        
        if is_vlm:
            if not VLM_AVAILABLE:
                raise ImportError(
                    f"Qwen3VLForConditionalGeneration is not available in your transformers version. "
                    f"Please upgrade transformers to use VLM models: pip install --upgrade transformers"
                )
            logger.warning("VLM models with Quanto quantization - note that quantization may not be fully supported for VLM architectures")
        
        if config.identifier.is_local:
            return self._load_local_model(
                config=config,
                compile_mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                is_vlm=is_vlm
            )

        # For HF models, use cache-based approach
        bits = config.identifier.bits.value if config.identifier.bits else 8
        cache_path = self._get_cache_path(config.model_path, bits, is_vlm)
        logger.info(f"Loading {bits}-bit Quanto {'VLM' if is_vlm else ''} model with cache path: {cache_path}")

        # Try loading from cache first
        cached_model = self._load_quanto_from_cache(cache_path, config, is_vlm)
        if cached_model is not None:
            model, tokenizer_or_processor = cached_model
            
            # Apply compilation if needed (Quanto is torch.compile friendly)
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
                    logger.info(f"Compiling cached Quanto model with mode: {compile_mode}")
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
        """Load a locally quantized QUANTO model using optimum.quanto."""
        logger.debug(f"Loading local QUANTO {'VLM' if is_vlm else ''} model from: {config.model_path}")
        
        # Verify required files exist
        safetensors_path = os.path.join(config.model_path, "model.safetensors")
        quantmap_path = os.path.join(config.model_path, "quantization_map.json")
        
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Model weights not found at: {safetensors_path}")
        if not os.path.exists(quantmap_path):
            raise FileNotFoundError(f"Quantization map not found at: {quantmap_path}")
        
        # Load model components
        state_dict = load_file(safetensors_path)
        with open(quantmap_path, 'r') as f:
            quantization_map = json.load(f)
        
        cache_dir = self.standard_vlm_cache_dir if is_vlm else self.standard_cache_dir

        # Initialize model
        model_config = AutoConfig.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=config.trust_remote_code,
            cache_dir=cache_dir
        )
        
        with init_empty_weights():
            if is_vlm:
                model = Qwen3VLForConditionalGeneration.from_config(
                    model_config,
                    trust_remote_code=config.trust_remote_code
                )
            else:
                model = AutoModelForCausalLM.from_config(
                    model_config, 
                    trust_remote_code=config.trust_remote_code
                )
        
        # Apply quantization
        from optimum.quanto import requantize
        requantize(model, state_dict, quantization_map, config.device)
        
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
        """Load a QUANTO model directly from HuggingFace using transformers."""
        logger.debug(f"Loading HuggingFace QUANTO {'VLM' if is_vlm else ''} model: {config.model_path}")
        
        # Determine quantization bit width from config
        bits = config.identifier.bits.value if config.identifier.bits else 8
        quanto_config = QuantoConfig(weights=f"int{bits}")
        
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
            logger.info(f"Loading model {config.model_path} across multiple devices using max_memory configuration")
            if is_vlm:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory=max_memory,
                    quantization_config=quanto_config,
                    trust_remote_code=config.trust_remote_code,
                    cache_dir=cache_dir
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory=max_memory,
                    quantization_config=quanto_config,
                    trust_remote_code=config.trust_remote_code,
                    cache_dir=cache_dir
                )
            # Skip compilation for multi-GPU models as it's not compatible
        else:
            logger.info(f"Loading model {config.model_path} on single device: {config.device}")
            if is_vlm:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16,
                    device_map=config.device,
                    max_memory=max_memory,
                    quantization_config=quanto_config,
                    trust_remote_code=config.trust_remote_code,
                    cache_dir=cache_dir
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16,
                    device_map=config.device,
                    max_memory=max_memory,
                    quantization_config=quanto_config,
                    trust_remote_code=config.trust_remote_code,
                    cache_dir=cache_dir
                )
            
            # Compile model if CUDA is available (only for single-GPU case)
            if torch.cuda.is_available() and config.apply_compile:
                logger.info(f"Compiling model with mode: {compile_mode}")
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

        # Save to cache using Quanto's serialization workflow
        try:
            logger.info(f"Saving Quanto model to cache: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Freeze the model (convert to static quantization)
            from optimum.quanto import freeze, quantization_map
            freeze(model)
            
            # Save model state_dict as safetensors
            safetensors_path = cache_path / "model.safetensors"
            save_file(model.state_dict(), str(safetensors_path))
            
            # Save quantization map as JSON
            quantmap_path = cache_path / "quantization_map.json"
            with open(quantmap_path, 'w') as f:
                json.dump(quantization_map(model), f)
            
            # Save tokenizer or processor
            tokenizer_or_processor.save_pretrained(cache_path)
            
            logger.info(f"Successfully saved Quanto model to cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save Quanto model to cache: {str(e)}")
            logger.warning("Continuing without caching...")

        return model, tokenizer_or_processor
    
    def _load_tokenizer(self, config: ModelConfig) -> AutoTokenizer:
        """Load tokenizer with common configuration."""
        return AutoTokenizer.from_pretrained(
            config.tokenizer_path,
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
