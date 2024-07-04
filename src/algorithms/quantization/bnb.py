import os
import time
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from src import MODEL_SAVE_PATH

bnb_base_config = {
    "llm_int8_threshold": 6.0,
    "llm_int8_enable_fp32_cpu_offload": False,
    "llm_int8_has_fp16_weight": False,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "fp4",
    "bnb_4bit_use_double_quant": False,
}

def quantize_bnb(model_name, num_bits=None, save_model=False, save_path="", device="cuda"):
    if num_bits is None or num_bits not in [4, 8]:
        raise ValueError(f"Invalid num_bits for BNB: {num_bits}")

    bnb_config = BitsAndBytesConfig()
    if num_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=bnb_base_config["llm_int8_threshold"],
            llm_int8_enable_fp32_cpu_offload=bnb_base_config["llm_int8_enable_fp32_cpu_offload"],
            llm_int8_has_fp16_weight=bnb_base_config["llm_int8_has_fp16_weight"],
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_base_config["bnb_4bit_compute_dtype"],
            bnb_4bit_quant_type=bnb_base_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=bnb_base_config["bnb_4bit_use_double_quant"],
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

    bnb_model_name = f"{model_name.split('/')[1]}-bnb-{num_bits}bit"
    bnb_model_path = os.path.join(MODEL_SAVE_PATH, bnb_model_name)
    os.makedirs(bnb_model_path, exist_ok=True)
    bnb_model.PATH = bnb_model_path
    
    accelerator = Accelerator()
    accelerator.save_model(bnb_model, bnb_model_path)
    bnb_model.NAME = bnb_model_name
    
    print(f'Model is quantized and saved at "{bnb_model_path}"')
    
    # Calculate model size and GPU utilization
    # calculate_model_size(bnb_model_path)
    # from src.models.utils_llm import print_gpu_utilization
    # print_gpu_utilization()

    if save_model:
        save_dir = save_path if save_path else bnb_model_path
        bnb_model.save_pretrained(save_dir)

    return bnb_model
