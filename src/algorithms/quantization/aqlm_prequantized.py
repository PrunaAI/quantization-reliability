import time
import torch
from transformers import AutoModelForCausalLM

from src.algorithms.quantization.config.aqlm_config import AQLM_MODEL_VARIANTS

import logging
logger = logging.getLogger("quant_logger")

def quantize_aqlm_prequantized(quantize_config={}, device="cuda"):
    if quantize_config['num_bits'] is None or quantize_config['num_bits'] not in [1, 2]:
        raise ValueError(f"Invalid num_bits for BNB: {quantize_config['num_bits']}")
    
    if 'model_variant' not in quantize_config:
        raise ValueError("Missing required key 'model_variant' in quantize_config")
    
    # Validate the model_variant
    if quantize_config['model_variant'] not in AQLM_MODEL_VARIANTS:
        raise ValueError(f"Invalid model_variant: {quantize_config['model_variant']}")

    # Load the tokenizer and model
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(quantize_config['model_variant'], torch_dtype="auto", device_map=device)
    end_time = time.time()  # End time measurement
    model.QUANT_TIME = end_time - start_time
    logger.info(f"Quantization time: {model.QUANT_TIME:.2f} seconds")
    
    # Add additional attributes to the model
    model.NAME = quantize_config['model_variant']
    
    logger.info(f"Model {quantize_config['model_variant']} is loaded.")
    
    return model

