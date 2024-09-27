from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.error import HTTPError
from typing import Tuple, Optional

import os
import torch
from typing import Tuple, Optional
import logging

from hqq.engine.hf import HQQModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from awq import AutoAWQForCausalLM

from src.models import base_models, hf_quantized_models, local_quantized_models, local_tokenizers
logger = logging.getLogger("quant_logger")


from src import MODEL_SAVE_PATH

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

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.

    Raises:
        ValueError: If the model name is not found or the local model is invalid.
        OSError: If there's an error loading the model or tokenizer.
    """
    try:
        logger.info(f"Attempting to load model: {model_name}")
        
        # Determine the actual model path
        if model_name in local_quantized_models:
            model_path = local_quantized_models[model_name]
            is_local_model = True
        elif model_name in hf_quantized_models:
            model_path = hf_quantized_models[model_name]
            is_local_model = False
        elif model_name in base_models:
            model_path = base_models[model_name]
            is_local_model = False
        else:
            raise ValueError(f"Model {model_name} not found in any of the model dictionaries")

        if is_local_model:
            logger.info(f"Loading locally quantized model from: {model_path}")
            if not os.path.exists(model_path):
                raise OSError(f"Local model path does not exist: {model_path}")

        # Special handling for HQQ models
        if "HQQ" in model_name:
            try:
                model = HQQModelForCausalLM.from_quantized(model_path, device_map='auto', cache_dir=cache_dir)
            except:
                model = AutoHQQHFModel.from_quantized(model_path, device_map='auto', cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizers[model_name] if is_local_model else model_path, device_map='auto', cache_dir=cache_dir)

        # Special handling for AWQ model from PrunaAI
        elif "AWQ" in model_name:
            model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto', cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizers[model_name] if is_local_model else model_path, device_map='auto', cache_dir=cache_dir)
            model.dtype = torch.float16
        else:
            # Load from Hugging Face or local path
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device,
                max_memory=max_memory,
                cache_dir=cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizers[model_name] if is_local_model else model_path, device_map='auto', cache_dir=cache_dir)
        
        model.NAME = model_name
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        if tokenizer.model_max_length > 1e6:
            logger.warning(f"Tokenizer model max length reduced from {tokenizer.model_max_length} to 2048 to fit in memory")
            tokenizer.model_max_length = 2048
        
        logger.info(f"Successfully loaded model: {model_name}")
        logger.info(f"Model configuration:")
        logger.info(f"Model cache directory: {model.config.cache_dir}")
        logger.info(f"- Model max length: {tokenizer.model_max_length}")
        logger.info(f"- Model dtype: {getattr(model, 'dtype', 'Not available')}")
        logger.info(f"- Model device: {getattr(model, 'device', 'Not available')}")
        logger.info(f"- Model parameters: {model.num_parameters():,}")
        logger.info(f"- Model memory footprint: {model.get_memory_footprint() / (1024 ** 3):.2f} GB")
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
