import os
import time
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from src import MODEL_SAVE_PATH

import logging
logger = logging.getLogger("quant_logger")

def quantize_bnb(model_name, quantize_config={}, save_model=False, save_path="", device="cuda"):
    if quantize_config['num_bits'] is None or quantize_config['num_bits'] not in [4, 8]:
        raise ValueError(f"Invalid num_bits for BNB: {quantize_config['num_bits']}")

    bnb_config = BitsAndBytesConfig()
    if quantize_config['num_bits'] == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=quantize_config["load_in_8bit"],
            llm_int8_threshold=quantize_config["llm_int8_threshold"],
            llm_int8_enable_fp32_cpu_offload=quantize_config["llm_int8_enable_fp32_cpu_offload"],
            llm_int8_has_fp16_weight=quantize_config["llm_int8_has_fp16_weight"],
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantize_config["load_in_4bit"],
            bnb_4bit_compute_dtype=quantize_config["bnb_4bit_compute_dtype"],
            bnb_4bit_quant_type=quantize_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quantize_config["bnb_4bit_use_double_quant"],
        )
    
    start_time = time.time()
    bnb_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        torch_dtype=torch.float32,
        device_map=device
    )
    end_time = time.time()  # End time measurement
    bnb_model.QUANT_TIME = end_time - start_time
    logger.info(f"Quantization time: {bnb_model.QUANT_TIME:.2f} seconds")

    bnb_model_name = f"{model_name.split('/')[1]}-{quantize_config['name']}"
    bnb_model_path = os.path.join(MODEL_SAVE_PATH, bnb_model_name)
    os.makedirs(bnb_model_path, exist_ok=True)
    bnb_model.PATH = bnb_model_path
    
    accelerator = Accelerator()
    accelerator.save_model(bnb_model, bnb_model_path)
    bnb_model.NAME = bnb_model_name
    
    logger.info(f'Model {model_name} is quantized to {bnb_model_name} and saved at "{bnb_model_path}"')
    
    if save_model:
        save_dir = save_path if save_path else bnb_model_path
        bnb_model.save_pretrained(save_dir)

    return bnb_model
