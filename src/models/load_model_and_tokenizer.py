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
logger = logging.getLogger("quant_logger")


def load_model_and_tokenizer(
    model_name: str,
    quantization_method: Optional[str] = None,
    device: str = "cuda",
    max_memory: Optional[dict] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from Hugging Face or a locally quantized model.

    Args:
        model_name (str): Name of the model to load.
        quantization_method (str, optional): Quantization method used for local models.
        device (str): Device to load the model on. Defaults to "cuda".
        max_memory (dict, optional): Maximum memory to use for model loading.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.

    Raises:
        ValueError: If the model name is not found or the quantization method is invalid.
        OSError: If there's an error loading the model or tokenizer.
    """
    try:
        logger.info(f"Attempting to load model: {model_name}")
        
        # Check if it's a locally quantized model
        if quantization_method:
            if model_name not in local_quantized_models:
                raise ValueError(f"No local path found for model: {model_name}")
            
            model_path = local_quantized_models[model_name]
            logger.info(f"Loading locally quantized model from: {model_path}")
            
            if not os.path.exists(model_path):
                raise OSError(f"Local model path does not exist: {model_path}")

        # Special handling for HQQ models
        if "HQQ" in model_name:
            try:
                model = HQQModelForCausalLM.from_quantized(model_name, device_map='auto')
            except:
                model = AutoHQQHFModel.from_quantized(model_name)
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

        # Special handling for AWQ model from PrunaAI
        elif "AWQ" in model_name and "PrunaAI" in model_name:
            model = AutoAWQForCausalLM.from_quantized(model_name, trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

        else:
            # Load from Hugging Face or local path
            model = AutoModelForCausalLM.from_pretrained(
                model_name if not quantization_method else model_path,
                torch_dtype="auto",
                device_map=device,
                max_memory=max_memory
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name if not quantization_method else model_path)
        
        model.NAME = model_name
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        if tokenizer.model_max_length > 1e6:
            logger.warning(f"Tokenizer model max length reduced from {tokenizer.model_max_length} to 2048 to fit in memory")
            tokenizer.model_max_length = 2048
        
        logger.info(f"Successfully loaded model: {model_name}")
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