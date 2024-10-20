import os
import torch
import logging
import re


from src import MODEL_SAVE_PATH
from src.algorithms.quantization.config import QUANT_CONFIGS
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
    "Llama-3-8B-AQLM-2bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16",
    "Llama-3-8B-AQLM-PV-2bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16",
    "Llama-3-8B-AQLM-PV-1bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16",
    
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
    "Llama-3-8B-AWQ-4bit-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AWQ-4"),
    
    # BNB models
    "Llama-3-8B-BNB-8bit-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-8"),
    "Llama-3-8B-BNB-4bit-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-4"),
    
    # HQQ models
    "Llama-3-8B-HQQ-8-uniform-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-8-uniform"),
    "Llama-3-8B-HQQ-mixed-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-mixed"),
    
    # QUANTO models
    "Llama-3-8B-QUANTO-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO"),
    "Llama-3-8B-QUANTO-CALIB-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-CALIB"),
    "Llama-3-8B-QUANTO-QAT-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-QAT"),
    
    # HQQ-LORA models
    "Llama-3-8B-HQQ-LORA-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-LORA"),
    
    # AQLM-LORA models
    "Llama-3-8B-AQLM-LORA-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AQLM-LORA"),
}

META_LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
local_tokenizers = {
    "Llama-3-8B-AWQ-4bit-local": META_LLAMA_3_8B,
    "Llama-3-8B-BNB-8bit-local": META_LLAMA_3_8B,
    "Llama-3-8B-BNB-4bit-local": META_LLAMA_3_8B,
    "Llama-3-8B-HQQ-8-uniform-local": META_LLAMA_3_8B,
    "Llama-3-8B-HQQ-mixed-local": META_LLAMA_3_8B,
    "Llama-3-8B-QUANTO-local": META_LLAMA_3_8B,
    "Llama-3-8B-QUANTO-CALIB-local": META_LLAMA_3_8B,
    "Llama-3-8B-QUANTO-QAT-local": META_LLAMA_3_8B,
    "Llama-3-8B-HQQ-LORA-local": META_LLAMA_3_8B,
    "Llama-3-8B-AQLM-LORA-local": META_LLAMA_3_8B,
}

LLAMA_3_8B_MODEL_TO_CONFIG_MAP = {
    "Llama-3-8B": "NONE",
    "Llama-3-8B-BNB-4bit-local": "BNB-4",
    "Llama-3-8B-BNB-8bit-local": "BNB-8",
    "Llama-3-8B-AWQ-4bit-local": "AWQ-4",
    "Llama-3-8B-HQQ-8-uniform-local": "HQQ-8-uniform",
    "Llama-3-8B-HQQ-mixed-local": "HQQ-mixed",
    "Llama-3-8B-HQQ-LORA-local": "HQQ-LORA",
    "Llama-3-8B-QUANTO-local": "QUANTO",
    "Llama-3-8B-QUANTO-CALIB-local": "QUANTO-CALIB",
    "Llama-3-8B-QUANTO-QAT-local": "QUANTO-QAT",
    "Llama-3-8B-AQLM-local": "AQLM",
    "Llama-3-8B-AQLM-LORA-local": "AQLM-LORA",
}

def get_model_num_bits(model_name):
    """
    Get the number of bits used for quantization of a given model.

    Args:
    model_name (str): The name of the model (e.g., 'Llama-3-8B-AWQ-4bit-local', 'Llama-3-8B-AQLM-2bit', 'Llama-3-8B')

    Returns:
    int: The number of bits used for quantization or storage

    Raises:
    ValueError: If the model name is not recognized or if num_bits cannot be determined
    """
    # Check if it's a locally quantized model
    if model_name in LLAMA_3_8B_MODEL_TO_CONFIG_MAP:
        config_short_name = LLAMA_3_8B_MODEL_TO_CONFIG_MAP[model_name]
        config_dict = QUANT_CONFIGS[config_short_name]
        num_bits = config_dict.get('num_bits')
        if num_bits is not None:
            return int(num_bits)
    
    # Check if it's a Hugging Face quantized model
    if model_name in hf_quantized_models:
        match = re.search(r'(\d+)bit', model_name)
        if match:
            return int(match.group(1))
    
    # Check if it's a base model
    if model_name in base_models:
        # Base models typically use 32-bit (float32) or 16-bit (float16) precision
        base_model_bits = {
            "TinyLlama-Chat": 16,  # Assuming TinyLlama models use 16-bit precision
            "TinyLlama": 16,
            "Llama-3-8B": 16,  # Assuming Llama-3 uses 16-bit precision
            "Bloomz": 16,      # Assuming Bloomz uses 16-bit precision
            "GPT2-Large": 32   # GPT-2 typically uses 32-bit precision
        }
        return base_model_bits.get(model_name, 32)  # Default to 32 if not specified
    
    # If we've reached here, we couldn't determine the number of bits
    raise ValueError(f"Unable to determine number of bits for model: {model_name}")