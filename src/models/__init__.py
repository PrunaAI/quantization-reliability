import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.error import HTTPError
import re

# Define the base datasets dictionary
base_models = {
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
}

def get_model(model_name=None, directory_model=None, cache_dir=None, seed=123, device="cuda"):
    torch.manual_seed(seed)
    if model_name is not None:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        model.NAME = model_name
    elif model_name is None and directory_model is not None:
        # Load model from local directory
        try:
            tokenizer = AutoTokenizer.from_pretrained(directory_model)
            model = AutoModelForCausalLM.from_pretrained(directory_model)
            model.NAME = model_name
        except (OSError, HTTPError) as e:
            print(f"Error loading model from directory: {directory_model}")
            print(f"Error message: {e}")
    else:
        # No model name or directory provided, raise an error
        raise ValueError("Please specify either model_name or directory_model")
    
    return model, tokenizer

def get_model_name(model_name):
    try:
        return base_models[model_name]
    except KeyError:
        raise NotImplementedError(f"Model {model_name} is not spelled correctly or not yet supported")
    
