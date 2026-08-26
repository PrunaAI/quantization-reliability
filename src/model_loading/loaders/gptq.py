from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, GPTQConfig
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import logging
from src.config import HF_CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.loaders.interface import ModelLoaderInterface
from src.model_loading.common.model_enums import BitPrecision, ModelFamily, ModelSize

logger = logging.getLogger("reliability_eval")

# Conditional import for VLM models
try:
    from transformers import Qwen3VLForConditionalGeneration
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None

class GPTQModelLoader(ModelLoaderInterface):
    """
    Merged loader for GPTQ quantized models that uses:
    - gptqmodel package for Llama-3.2 3B models with 4 or 8 bits
    - HuggingFace Transformers for all other models
    Supports both standard models and Vision-Language Models.
    """
    
    def __init__(self):
        # Cache directory for specialized gptqmodel package
        self.gptqmodel_cache_dir = Path(HF_CACHE_ROOT) / "gptq_models"
        self.gptqmodel_cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache directory for HuggingFace transformers
        self.hf_cache_dir = Path(HF_CACHE_ROOT) / "gptq_models_hf"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        # Separate cache for VLM models
        self.vlm_cache_dir = Path(HF_CACHE_ROOT) / "gptq_vlm_models"
        self.vlm_cache_dir.mkdir(parents=True, exist_ok=True)

        # Shared base model cache — reused across all quantization methods
        self.standard_cache_dir = Path(HF_CACHE_ROOT) / "standard_models"
        self.standard_cache_dir.mkdir(parents=True, exist_ok=True)

        self.standard_vlm_cache_dir = Path(HF_CACHE_ROOT) / "vlm_models"
        self.standard_vlm_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_vlm_model(self, config: ModelConfig) -> bool:
        """Check if the model is a Vision-Language Model"""
        return config.identifier.family in [ModelFamily.QWEN3VL, ModelFamily.QWEN25VL]
    
    def _should_use_gptqmodel(self, config: ModelConfig) -> bool:
        """Determine whether to use gptqmodel package or HuggingFace."""
        # Set to True to use GPTQModel for all bits and models
        is_opt = config.identifier.family == ModelFamily.OPT
        if is_opt:
            conditions = [
                config.identifier.bits in [],
            ]
            return all(conditions)
        
        is_qwen3 = (config.identifier.family in [ModelFamily.QWEN3, ModelFamily.QWEN3VL, ModelFamily.QWEN25VL])
        if is_qwen3:
            conditions = [
                config.identifier.bits in [BitPrecision.INT2, BitPrecision.INT3, BitPrecision.INT4, BitPrecision.INT8],
            ]
            return all(conditions)

        conditions = [
            config.identifier.bits in [BitPrecision.INT2, BitPrecision.INT3, BitPrecision.INT4, BitPrecision.INT8]
        ]
        return all(conditions)
    
    def get_model_cache_paths(self, config):
        is_vlm = self._is_vlm_model(config)
        use_gptqmodel = self._should_use_gptqmodel(config)
        hf_base_path = (self.standard_vlm_cache_dir if is_vlm else self.standard_cache_dir) / f"models--{config.model_path.replace('/', '--')}"
        quantized_path = self._get_cache_path(config.model_path, config.identifier.bits.value, use_gptqmodel, is_vlm)
        return [hf_base_path, quantized_path]

    def _get_cache_path(self, model_path: str, bits: int, use_gptqmodel: bool, is_vlm: bool = False) -> Path:
        """Generate unique cache path for model based on name, bits, and implementation."""
        model_name = model_path.split('/')[-1]
        
        if is_vlm:
            cache_dir = self.vlm_cache_dir
            suffix = "vlm"
        else:
            cache_dir = self.gptqmodel_cache_dir if use_gptqmodel else self.hf_cache_dir
            suffix = "gptqmodel" if use_gptqmodel else "hf"
        
        return cache_dir / f"{model_name}_{bits}bit_{suffix}"
    
    #------------------- HuggingFace Transformers implementation -------------------#
    
    def _load_hf_from_cache(
        self,
        cache_path: Path,
        config: ModelConfig,
        is_vlm: bool = False
    ) -> Optional[Union[
        Tuple[AutoModelForCausalLM, AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]]:
        """Try loading HuggingFace model from cache path."""
        if not cache_path.exists():
            return None
        try:
            logger.info(f"Loading HF GPTQ {'VLM' if is_vlm else ''} model from cache: {cache_path}")
            
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
            
            # Set device_map based on multi-GPU configuration
            device_map = "auto" if is_multi_gpu else config.device
            logger.info(f"Loading HF GPTQ model with device_map={device_map}")
            
            if is_vlm:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    cache_path,
                    device_map=device_map,
                    max_memory=max_memory,
                    trust_remote_code=config.trust_remote_code
                )
                processor = AutoProcessor.from_pretrained(
                    config.tokenizer_path or config.model_path,
                    cache_dir=self.vlm_cache_dir,
                    trust_remote_code=config.trust_remote_code
                )
                return model, processor
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cache_path,
                    device_map=device_map,
                    max_memory=max_memory,
                    trust_remote_code=config.trust_remote_code
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    config.tokenizer_path or config.model_path,
                    trust_remote_code=config.trust_remote_code
                )
                return model, tokenizer
        except Exception as e:
            logger.warning(f"Failed to load from HF cache: {e}")
            return None
    
    def _quantize_and_save_hf(
        self,
        config: ModelConfig,
        cache_path: Path,
        is_vlm: bool = False
    ) -> Union[
        Tuple[AutoModelForCausalLM, AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]:
        """Quantize HuggingFace model and save to cache."""
        logger.info(f"Quantizing HF GPTQ {'VLM' if is_vlm else ''} model and saving to: {cache_path}")
        
        cache_dir = self.standard_vlm_cache_dir if is_vlm else self.standard_cache_dir

        if is_vlm:
            processor = AutoProcessor.from_pretrained(
                config.tokenizer_path or config.model_path,
                cache_dir=cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            gptq_config = GPTQConfig(
                bits=config.identifier.bits.value,
                dataset="c4",
                tokenizer=processor.tokenizer  # Use the underlying tokenizer
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path or config.model_path,
                trust_remote_code=config.trust_remote_code
            )
            
            gptq_config = GPTQConfig(
                bits=config.identifier.bits.value,
                dataset="c4",
                tokenizer=tokenizer
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
        
        # Set device_map based on multi-GPU configuration
        device_map = "auto" if is_multi_gpu else config.device
        logger.info(f"Quantizing HF GPTQ model with device_map={device_map}")
        
        if is_vlm:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_path,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=config.trust_remote_code,
                quantization_config=gptq_config,
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=config.trust_remote_code,
                quantization_config=gptq_config,
                torch_dtype=torch.float16
            )
        
        model.save_pretrained(cache_path)
        
        if is_vlm:
            return model, processor
        else:
            return model, tokenizer
    
    #------------------- gptqmodel package implementation -------------------#
    
    def _load_calibration_dataset(self, samples: int = 512):
        """Load a small calibration dataset for quantization."""
        return load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00000-of-01024.json.gz",
            split="train"
        ).select(range(samples))["text"]
    
    def _move_non_quantized_to_device(self, model: GPTQModel, device: str) -> GPTQModel:
        """Move embedding and lm_head to target device — GPTQ only quantizes linear layers."""
        inner = model.model  # the HF CausalLM wrapper
        # embed_tokens lives inside the base model (e.g. model.model.model.embed_tokens)
        base = getattr(inner, "model", None)
        if base is not None and hasattr(base, "embed_tokens"):
            base.embed_tokens = base.embed_tokens.to(device)
        if hasattr(inner, "lm_head"):
            inner.lm_head = inner.lm_head.to(device)
        return model

    def _compile_model(self, model: GPTQModel, compile_mode: str, fullgraph: bool, dynamic: bool) -> GPTQModel:
        """Apply torch.compile to speed up model inference."""
        logger.info(f"Compiling model with mode={compile_mode}, fullgraph={fullgraph}, dynamic={dynamic}")
        try:
            # Apply torch.compile to the underlying pytorch model
            model.model = torch.compile(
                model.model, 
                mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Continuing with uncompiled model.")
        return model
    
    def _load_gptqmodel_from_cache(self, cache_path: Path, config: ModelConfig) -> Optional[Tuple[GPTQModel, AutoTokenizer]]:
        """Try loading gptqmodel model from cache path."""
        if not cache_path.exists():
            return None
        try:
            logger.info(f"Loading gptqmodel GPTQ model from cache: {cache_path}")
            model = GPTQModel.load(
                str(cache_path),
                device=config.device
            )
            self._move_non_quantized_to_device(model, config.device)
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path or config.model_path,
                trust_remote_code=config.trust_remote_code
            )
            return model, tokenizer
        except Exception as e:
            logger.warning(f"Failed to load from gptqmodel cache: {e}")
            return None
    
    def _quantize_and_save_gptqmodel(self, config: ModelConfig, cache_path: Path) -> Tuple[GPTQModel, AutoTokenizer]:
        """Quantize model with gptqmodel and save to cache."""
        logger.info(f"Quantizing gptqmodel GPTQ model to {config.identifier.bits.value} bits and saving to: {cache_path}")
        
        # Configure quantization settings
        quant_config = QuantizeConfig(
            bits=config.identifier.bits.value,
            group_size=128,
            desc_act=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model_path,
            trust_remote_code=config.trust_remote_code
        )
        
        # Load model with quantization config
        model = GPTQModel.load(
            config.model_path,
            quant_config,
            device=config.device
        )
        
        # Load calibration dataset and quantize the model
        calibration_dataset = self._load_calibration_dataset()
        model.quantize(
            calibration_dataset,
            batch_size=1
        )
        
        # Save the quantized model
        model.save(str(cache_path))
        self._move_non_quantized_to_device(model, config.device)

        return model, tokenizer
    
    #------------------- Main load_model method -------------------#
    
    def load_model(
        self,
        config: ModelConfig,
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False
    ) -> Union[
        Tuple[Union[GPTQModel, AutoModelForCausalLM], AutoTokenizer],
        Tuple[object, AutoProcessor]
    ]:
        """
        Load GPTQ model with automatic selection between gptqmodel and HuggingFace
        implementations based on model characteristics. Supports VLM models.
        """
        if not config.identifier.bits:
            raise ValueError("Bits must be specified for GPTQ quantization")
            
        if config.identifier.bits not in [BitPrecision.INT2, BitPrecision.INT3, BitPrecision.INT4, BitPrecision.INT8]:
            raise ValueError(f"GPTQ supports 2,3,4,8 bit quantization, got: {config.identifier.bits}")
        
        # Check if this is a VLM model
        is_vlm = self._is_vlm_model(config)
        
        if is_vlm:
            if not VLM_AVAILABLE:
                raise ImportError(
                    f"Qwen3VLForConditionalGeneration is not available in your transformers version. "
                    f"Please upgrade transformers to use VLM models: pip install --upgrade transformers"
                )
            logger.warning("VLM models with GPTQ quantization - note that quantization may not be fully supported for VLM architectures")
        
        # Determine which implementation to use
        use_gptqmodel = self._should_use_gptqmodel(config)
        cache_path = self._get_cache_path(config.model_path, config.identifier.bits.value, use_gptqmodel, is_vlm)
        
        logger.info(f"Loading {config.identifier.bits.value}-bit GPTQ {'VLM' if is_vlm else ''} model using {'gptqmodel' if use_gptqmodel else 'HuggingFace'} implementation")
        
        if use_gptqmodel:
            # Try loading from cache first
            cached_model = self._load_gptqmodel_from_cache(cache_path, config)
            if cached_model is not None:
                model, tokenizer = cached_model
                
                # Apply torch.compile if available
                if torch.__version__ >= "2.0.0"  and config.apply_compile:
                    model = self._compile_model(model, compile_mode, fullgraph, dynamic)
                else:
                    logger.warning("torch.compile requires PyTorch 2.0+. Skipping compilation.")
                
                return model, tokenizer
            
            # If not in cache, quantize and save
            model, tokenizer = self._quantize_and_save_gptqmodel(config, cache_path)
            
            # Apply torch.compile if available
            if torch.__version__ >= "2.0.0" and config.apply_compile:
                model = self._compile_model(model, compile_mode, fullgraph, dynamic)
            else:
                logger.warning("torch.compile requires PyTorch 2.0+. Skipping compilation.")
            
            return model, tokenizer
        else:
            # Try loading from cache first
            cached_model = self._load_hf_from_cache(cache_path, config, is_vlm)
            if cached_model is not None:
                model, tokenizer_or_processor = cached_model
                
                # Apply torch.compile if available
                if torch.__version__ >= "2.0.0":
                    try:
                        # Check for multi-GPU configuration
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
                        
                        # Skip compilation for multi-GPU models
                        if not is_multi_gpu and config.apply_compile:
                            logger.info(f"Compiling HF model with mode={compile_mode}, fullgraph={fullgraph}, dynamic={dynamic}")
                            model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph, dynamic=dynamic)
                            logger.info("HF Model compilation successful")
                        else:
                            logger.info("Skipping compilation for multi-GPU model")
                    except Exception as e:
                        logger.warning(f"HF Model compilation failed: {e}. Continuing with uncompiled model.")
                else:
                    logger.warning("torch.compile requires PyTorch 2.0+. Skipping compilation.")
                
                return model, tokenizer_or_processor
            
            # If not in cache, quantize and save
            model, tokenizer_or_processor = self._quantize_and_save_hf(config, cache_path, is_vlm)
            
            # Apply torch.compile if available
            if torch.__version__ >= "2.0.0":
                try:
                    # Check for multi-GPU configuration
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
                    
                    # Skip compilation for multi-GPU models
                    if not is_multi_gpu and config.apply_compile:
                        logger.info(f"Compiling HF model with mode={compile_mode}, fullgraph={fullgraph}, dynamic={dynamic}")
                        model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph, dynamic=dynamic)
                        logger.info("HF Model compilation successful")
                    else:
                        logger.info("Skipping compilation for multi-GPU model")
                except Exception as e:
                    logger.warning(f"HF Model compilation failed: {e}. Continuing with uncompiled model.")
            else:
                logger.warning("torch.compile requires PyTorch 2.0+. Skipping compilation.")
            
            return model, tokenizer_or_processor
