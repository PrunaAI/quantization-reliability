import time
import torch
from transformers import AutoModelForCausalLM

def quantize_aqlm_prequantized(quantize_config={}, device="cuda"):
    if 'model_variant' not in quantize_config:
        raise ValueError("Missing required key 'model_variant' in quantize_config")
    
    # Validate the model_variant
    if quantize_config['model_variant'] not in quantize_config['model_variants']:
        raise ValueError(f"Invalid model_variant: {quantize_config['model_variant']}")

    # Load the tokenizer and model
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(quantize_config['model_variant'], torch_dtype="auto", device_map=device)
    end_time = time.time()  # End time measurement
    model.QUANT_TIME = end_time - start_time
    
    # Add additional attributes to the model
    model.NAME = quantize_config['model_variant']
    
    print(f"Model {quantize_config['model_variant']} is loaded.")
    
    return model

