from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from urllib.error import HTTPError
from typing import Tuple, Optional
from accelerate import init_empty_weights
from safetensors.torch import load_file
from optimum.quanto import (
    quantize,
    freeze,
    Calibration,
    requantize
)


import os
import io
import json
import torch
from typing import Tuple, Optional
import logging

from hqq.engine.hf import HQQModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from awq import AutoAWQForCausalLM

from src.models import BASE_MODELS, HF_QUANTIZED_MODELS, LLAMA_3_8B, LOCAL_QUANTIZED_MODELS, MODEL_TO_TOKENIZER_MAP
logger = logging.getLogger("quant_logger")

def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    max_memory: Optional[dict] = None,
    cache_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from Hugging Face or a locally quantized model.

    Args:
        model_name (str): Short name of the model to load.
        device (str): Device to load the model on. Defaults to "cuda".
        max_memory (dict, optional): Maximum memory to use for model loading.
        cache_dir (str, optional): Directory to cache the model.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.

    Raises:
        ValueError: If the model name is not found or the local model is invalid.
        OSError: If there's an error loading the model or tokenizer.
    """
    try:
        logger.info(f"Attempting to load model: {model_name}")
        model_path = None
        is_local_model = False
        model = None
        tokenizer = None
        
        # Determine the actual model path
        if model_name in LOCAL_QUANTIZED_MODELS:
            model_path = LOCAL_QUANTIZED_MODELS[model_name]
            is_local_model = True
        elif model_name in HF_QUANTIZED_MODELS:
            model_path = HF_QUANTIZED_MODELS[model_name]
            is_local_model = False
        elif model_name in BASE_MODELS:
            model_path = BASE_MODELS[model_name]
            is_local_model = False
        else:
            raise ValueError(f"Model {model_name} not found in any of the model dictionaries")

        # Special handling for QUANTO models
        if is_local_model and "QUANTO" in model_name:
            try:
                logger.info("Loading QUANTO quantized model...")
                
                # Get the base model path for config
                model_path = LOCAL_QUANTIZED_MODELS[model_name]
                base_model_path = LLAMA_3_8B
                state_dict_file = os.path.join(model_path, "model.safetensors")
                quantization_map_file = os.path.join(model_path, "quantization_map.json")
                
                # Load quantized weights
                state_dict = load_file(state_dict_file)
                with open(quantization_map_file, 'r') as f:
                    quantization_map = json.load(f)
                
                # Create an empty model from config
                model_reloaded = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
                config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, cache_dir=cache_dir)
                with init_empty_weights():
                    model_reloaded = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                    
                requantize(model_reloaded, state_dict, quantization_map, device)
                model = model_reloaded
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    device_map=device,
                    cache_dir=cache_dir
                )
                
            except Exception as e:
                logger.error(f"Error loading QUANTO model: {e}")
                raise

        # Special handling for HQQ models
        elif "HQQ" in model_name:
            try:
                model = HQQModelForCausalLM.from_quantized(model_path, device_map='auto', cache_dir=cache_dir)
            except:
                model = AutoHQQHFModel.from_quantized(model_path, device_map='auto', cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_TO_TOKENIZER_MAP[model_name] if is_local_model else model_path,
                device_map='cuda',
                cache_dir=cache_dir
            )

        # Special handling for AWQ model
        elif "AWQ" in model_name:
            model = AutoAWQForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device,
                cache_dir=cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_TO_TOKENIZER_MAP[model_name] if is_local_model else model_path,
                device_map=device,
                cache_dir=cache_dir
            )
            model.dtype = torch.float16
        else:
            logger.info(f"Loading locally quantized model from: {model_path}")
            if not os.path.exists(model_path):
                raise OSError(f"Local model path does not exist: {model_path}")
            
            # Load standard models from local path
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device,
                max_memory=max_memory,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_TO_TOKENIZER_MAP[model_name] if is_local_model else model_path,
                device_map=device,
                cache_dir=cache_dir
            )
        
        # Common post-processing for all models
        model.NAME = model_name
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        if tokenizer.model_max_length > 1e6:
            logger.warning(f"Tokenizer model max length reduced from {tokenizer.model_max_length} to 2048 to fit in memory")
            tokenizer.model_max_length = 2048
        
        # Log model information
        logger.info(f"Successfully loaded model: {model_name}")
        logger.info(f"Model configuration:")
        logger.info(f"Model cache directory: {getattr(model, 'cache_dir', 'Not available')}")
        logger.info(f"- Model max length: {tokenizer.model_max_length}")
        logger.info(f"- Model dtype: {getattr(model, 'dtype', 'Not available')}")
        logger.info(f"- Model device: {getattr(model, 'device', 'Not available')}")
        logger.info(f"- Model parameters: {getattr(model, 'num_parameters', 'Not available')}")
        logger.info(f"- Model memory footprint: {getattr(model, 'memory_footprint', 0) / (1024 ** 3):.2f} GB")
        logger.info(f"- Vocabulary size: {tokenizer.vocab_size}")
        logger.info(f"- Padding token ID: {tokenizer.pad_token_id}")
        logger.info(f"- Special tokens: {tokenizer.special_tokens_map}")
        
        return model, tokenizer
    
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except OSError as ose:
        logger.error(f"OSError: {ose}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading model: {e}")
        raise