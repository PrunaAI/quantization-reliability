from src.algorithms.quantization.config.aqlm_config import AQLM_LORA, AQLM
from src.algorithms.quantization.config.awq_config import AWQ_4
from src.algorithms.quantization.config.base_config import NONE
from src.algorithms.quantization.config.bnb_config import BNB_4, BNB_8
from src.algorithms.quantization.config.hqq_config import HQQ_LORA, HQQ_8_uniform, HQQ_mixed
from src.algorithms.quantization.config.quanto_config import QUANTO, QUANTO_CALIB, QUANTO_QAT
from src.models import base_models, hf_quantized_models, local_quantized_models

import re


DEBUG = True

QUANT_CONFIGS = {
    "NONE": NONE,
    "BNB-4": BNB_4,
    "BNB-8": BNB_8,
    "AWQ-4": AWQ_4,
    "HQQ-8-uniform": HQQ_8_uniform,
    "HQQ-mixed": HQQ_mixed,
    "HQQ-LORA": HQQ_LORA,
    "QUANTO": QUANTO,
    "QUANTO-CALIB": QUANTO_CALIB,
    "QUANTO-QAT": QUANTO_QAT,
    "AQLM": AQLM,
    "AQLM-LORA": AQLM_LORA,
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