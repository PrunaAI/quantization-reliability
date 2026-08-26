from typing import Tuple, Union
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    HqqConfig
)
from hqq.engine.hf import HQQModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
import logging
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.common.model_enums import BitPrecision, ModelFamily
from src.model_loading.loaders.interface import ModelLoaderInterface

logger = logging.getLogger("reliability_eval")

# Conditional import for VLM models
try:
    from transformers import Qwen3VLForConditionalGeneration
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None

class HQQModelLoader(ModelLoaderInterface):
    """Loader for HQQ quantized models supporting both local and HuggingFace loading, including VLM models."""
    
    def __init__(self):
        # Cache directory for HQQ models
        self.hqq_cache_dir = Path(HF_CACHE_ROOT) / "hqq_models"
        self.hqq_cache_dir.mkdir(parents=True, exist_ok=True)

        self.vlm_cache_dir = Path(HF_CACHE_ROOT) / "hqq_vlm_models"
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
        base_cache_dir = self.standard_vlm_cache_dir if self._is_vlm_model(config) else self.standard_cache_dir
        return [base_cache_dir / f"models--{config.model_path.replace('/', '--')}"]

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
        """Load model based on configuration type."""
        logger.info(f"Loading HQQ model with identifier: {config.identifier}")
        
        # Check if this is a VLM model
        is_vlm = self._is_vlm_model(config)
        
        if is_vlm:
            if not VLM_AVAILABLE:
                raise ImportError(
                    f"Qwen3VLForConditionalGeneration is not available in your transformers version. "
                    f"Please upgrade transformers to use VLM models: pip install --upgrade transformers"
                )
            logger.warning("VLM models with HQQ quantization - note that quantization may not be fully supported for VLM architectures")
        
        if config.identifier.is_local:
            return self._load_local_model(
                config=config,
                compile_mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                is_vlm=is_vlm
            )

        # Always quantize on-the-fly (no caching)
        bits = self._get_quantization_bits(config)
        logger.info(f"Quantizing {bits}-bit HQQ model on-the-fly (no cache)")

        return self._load_hf_model_no_cache(
            config=config,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            is_vlm=is_vlm
        )
        
    def _load_hf_model_no_cache(
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
        """Load and quantize HQQ model on-the-fly without caching."""
        logger.debug(f"Quantizing HuggingFace model on-the-fly: {config.model_path}")
        
        bits = self._get_quantization_bits(config)
        quant_config = self._create_quantization_config(bits)
        
        # Handle max_memory conversion
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
        cache_dir = self.standard_vlm_cache_dir if is_vlm else self.standard_cache_dir

        logger.info(f"Loading and quantizing {bits}-bit HQQ {'VLM' if is_vlm else ''} model: {config.model_path}")
        
        target_dtype = torch.float16 
        if is_vlm:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_path,
                torch_dtype=target_dtype,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=quant_config,
                trust_remote_code=config.trust_remote_code,
                cache_dir=cache_dir
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                torch_dtype=target_dtype,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=quant_config,
                trust_remote_code=config.trust_remote_code,
                cache_dir=cache_dir
            )
        
        # Compile if single GPU
        if torch.cuda.is_available() and config.apply_compile and not is_multi_gpu:
            logger.info(f"Compiling HQQ model with mode: {compile_mode}")
            model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph, dynamic=dynamic)
        
        # Load tokenizer or processor
        if is_vlm:
            processor = self._load_processor(config)
            return model, processor
        else:
            tokenizer = self._load_tokenizer(config)
            return model, tokenizer

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
        """Load a locally quantized HQQ model."""
        logger.debug(f"Loading local HQQ model from: {config.model_path}")
        
        cache_dir = self.vlm_cache_dir if is_vlm else self.hqq_cache_dir
        
        try:
            model = HQQModelForCausalLM.from_quantized(
                config.model_path,
                device_map='auto',
                cache_dir=cache_dir
            )
            
            # Compile model if CUDA is available
            if torch.cuda.is_available() and config.apply_compile:
                model = torch.compile(
                    model,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic
                )
        except Exception as e:
            logger.warning(f"Failed to load with HQQModelForCausalLM, attempting AutoHQQHFModel: {e}")
            model = AutoHQQHFModel.from_quantized(
                config.model_path,
                device_map='auto',
                cache_dir=cache_dir
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

    def _get_quantization_bits(self, config: ModelConfig) -> int:
        """Determine quantization bit width from config."""
        valid_bits = {
            BitPrecision.INT8.value,
            BitPrecision.INT4.value, 
            BitPrecision.INT3.value,
            BitPrecision.INT2.value,
            BitPrecision.INT1.value
        }
        default_bits = BitPrecision.INT4.value
        
        if not config.identifier.bits or config.identifier.bits in [BitPrecision.FP32, BitPrecision.FP16, BitPrecision.MIXED]:
            return default_bits
            
        bits = config.identifier.bits.value
        if bits not in valid_bits:
            logger.warning(f"Invalid bit width {bits}, defaulting to {default_bits}")
            return default_bits
        return bits

    def _create_quantization_config(self, bits: int, group_size: int = 64) -> HqqConfig:
        """Create HQQ configuration based on bit width."""
        return HqqConfig(
            nbits=bits,
            group_size=group_size,
            view_as_float=False,
            skip_modules=['lm_head'],
            compute_dtype=torch.float16
        )
