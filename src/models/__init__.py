from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.error import HTTPError
from typing import Tuple, Optional

import os
import torch
import logging


from src import MODEL_SAVE_PATH
logger = logging.getLogger("quant_logger")

# Base models dictionary
base_models = {
    # TinyLlama models
    "TinyLlama-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama": "TinyLlama/TinyLlama_v1.1",
    
    # Meta models
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    
    # BigScience models
    "Bloomz": "bigscience/bloomz-1b1",
    
    # OpenAI community models
    "GPT2-Large": "openai-community/gpt2-large",
}

# Hugging Face quantized models dictionary
hf_quantized_models = {
    # AQLM quantized models
    "Llama-3-8B-AQLM-2Bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16",
    "Llama-3-8B-AQLM-PV-2Bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16",
    "Llama-3-8B-AQLM-PV-1Bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16",
    
    # AWQ quantized models
    "Llama-3-8B-AWQ-4bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-AWQ-4bit-smashed",
    
    # BitsAndBytes (BNB) quantized models
    "Llama-3-8B-16K-bnb-4bit": "PrunaAI/mattshumer-Llama-3-8B-16K-bnb-4bit-smashed",
    
    # HQQ quantized models
    "Llama-3-8B-HQQ-4bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-4bit-smashed",
    "Llama-3-8B-HQQ-2bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-2bit-smashed",
    "Llama-3-8B-HQQ-1bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-1bit-smashed",
}

local_quantized_models = {
    # AWQ models
    "Llama-3-8B-AWQ": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AWQ"),
    
    # BNB models
    "Llama-3-8B-BNB-8bit": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-8bit"),
    "Llama-3-8B-BNB-4bit": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-4bit"),
    
    # HQQ models
    "Llama-3-8B-HQQ-8-uniform": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-8-uniform"),
    "Llama-3-8B-HQQ-mixed": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-mixed"),
    
    # QUANTO models
    "Llama-3-8B-QUANTO": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO"),
    "Llama-3-8B-QUANTO-CALIB": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-CALIB"),
    "Llama-3-8B-QUANTO-QAT": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-QAT"),
    
    # HQQ-LORA models
    "Llama-3-8B-HQQ-LORA": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-LORA"),
    
    # AQLM-LORA models
    "Llama-3-8B-AQLM-LORA": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AQLM-LORA"),
}

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


def get_model_path(model_name):
    if model_name in base_models:
        return base_models[model_name]
    elif model_name in hf_quantized_models:
        return hf_quantized_models[model_name]
    elif model_name in local_quantized_models:
        return local_quantized_models[model_name]
    else:
        raise NotImplementedError(f"Model {model_name} is not spelled correctly or not yet supported")
    
