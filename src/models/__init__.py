import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.error import HTTPError
import re

import logging
logger = logging.getLogger("quant_logger")

# Define the base datasets dictionary
base_models = {
    "TinyLlama-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "Bloomz": "bigscience/bloomz-1b1",
    "GPT2-Large": "openai-community/gpt2-large",
    "TinyLlama": "TinyLlama/TinyLlama_v1.1",
}

def get_model(model_name=None, directory_model=None, cache_dir=None, seed=123, device="cuda"):
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


def get_tokenizer(model_name=None, directory_model=None, cache_dir=None, seed=123, device="cuda"):
    logger.info(f"Loading tokenizer {model_name}")
    torch.manual_seed(seed)
    if model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, trust_remote_code=True)
    elif model_name is None and directory_model is not None:
        # Load tokenizer from local directory
        try:
            tokenizer = AutoTokenizer.from_pretrained(directory_model, trust_remote_code=True)
        except (OSError, HTTPError) as e:
            print(f"Error loading tokenizer from directory: {directory_model}")
            print(f"Error message: {e}")
    else:
        # No model name or directory provided, raise an error
        raise ValueError("Please specify either model_name or directory_model")
    
    return tokenizer


def get_model_name(model_name):
    try:
        return base_models[model_name]
    except KeyError:
        raise NotImplementedError(f"Model {model_name} is not spelled correctly or not yet supported")
    
