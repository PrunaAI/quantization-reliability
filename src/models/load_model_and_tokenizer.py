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

from src import MODEL_SAVE_PATH
from src.models import local_quantized_models
logger = logging.getLogger("quant_logger")


from src import MODEL_SAVE_PATH

def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "cuda",
    max_memory: Optional[dict] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from Hugging Face or a locally quantized model.

    Args:
        model_name_or_path (str): Name of the model to load from Hugging Face or path to local model.
        device (str): Device to load the model on. Defaults to "cuda".
        max_memory (dict, optional): Maximum memory to use for model loading.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.

    Raises:
        ValueError: If the model name is not found or the local model is invalid.
        OSError: If there's an error loading the model or tokenizer.
    """
    try:
        logger.info(f"Attempting to load model: {model_name_or_path}")
        
        # Check if it's a locally quantized model
        is_local_model = model_name_or_path.startswith(MODEL_SAVE_PATH)
        
        if is_local_model:
            logger.info(f"Loading locally quantized model from: {model_name_or_path}")
            
            if not os.path.exists(model_name_or_path):
                raise OSError(f"Local model path does not exist: {model_name_or_path}")

        # Special handling for HQQ models
        if "HQQ" in model_name_or_path:
            try:
                model = HQQModelForCausalLM.from_quantized(model_name_or_path, device_map='auto')
            except:
                model = AutoHQQHFModel.from_quantized(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(META_LLAMA_3_8B)

        # Special handling for AWQ model from PrunaAI
        elif "AWQ" in model_name_or_path:
            model = AutoAWQForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(META_LLAMA_3_8B)
        else:
            # Load from Hugging Face or local path
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype="auto",
                device_map=device,
                max_memory=max_memory
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        model.NAME = model_name_or_path
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        if tokenizer.model_max_length > 1e6:
            logger.warning(f"Tokenizer model max length reduced from {tokenizer.model_max_length} to 2048 to fit in memory")
            tokenizer.model_max_length = 2048
        
        logger.info(f"Successfully loaded model: {model_name_or_path}")
        logger.info(f"Model configuration:")
        logger.info(f"- Model max length: {tokenizer.model_max_length}")
        logger.info(f"- Model dtype: {model.dtype}")
        logger.info(f"- Model device: {model.device}")
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

def get_model(model_name=None, directory_model=None, seed=123, device="cuda"):
    logger.info(f"Loading model {model_name}")
    torch.manual_seed(seed)
    if model_name is not None:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)
        model.NAME = model_name
    elif model_name is None and directory_model is not None:
        # Load model from local directory
        try:
            model = AutoModelForCausalLM.from_pretrained(directory_model)
            model.NAME = model_name
        except (OSError, HTTPError) as e:
            print(f"Error loading model from directory: {directory_model}")
            print(f"Error message: {e}")
    else:
        # No model name or directory provided, raise an error
        raise ValueError("Please specify either model_name or directory_model")
    
    return model