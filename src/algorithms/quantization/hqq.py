import os
import time
import torch
from transformers import AutoModelForCausalLM, HqqConfig
from src import MODEL_SAVE_PATH

hqq_base_config = {
    "q3_group_size": 32,
    "q4_group_size": 64,
    "quant_zero": False,
    "quant_scale": False,
    "axis": 0
}

def quantize_hqq(model_name, num_bits=None, dynamic_config=False, save_model=False, save_path="", device="cuda"):
    if num_bits is None or num_bits not in [4, 8]:
        raise ValueError(f"Invalid num_bits for HQQ: {num_bits}")

    hqq_config = HqqConfig()
    # Option 1: All linear layers will use the same quantization config
    if dynamic_config:
        hqq_base_config_4 = {
            "nbits": 4,
            "group_size": hqq_base_config["q4_group_size"],
            "quant_zero": hqq_base_config["quant_zero"],
            "quant_scale": hqq_base_config["quant_scale"],
            "axis": hqq_base_config["axis"]
        }
        hqq_base_config_3 = {
            "nbits": 3,
            "group_size": hqq_base_config["q3_group_size"],
            "quant_zero": hqq_base_config["quant_zero"],
            "quant_scale": hqq_base_config["quant_scale"],
            "axis": hqq_base_config["axis"]
        }
        hqq_config = HqqConfig(dynamic_config={
            'self_attn.q_proj': hqq_base_config_4,
            'self_attn.k_proj': hqq_base_config_4,
            'self_attn.v_proj': hqq_base_config_4,
            'self_attn.o_proj': hqq_base_config_4,
            'mlp.gate_proj': hqq_base_config_3,
            'mlp.up_proj': hqq_base_config_3,
            'mlp.down_proj': hqq_base_config_3,
        })
    elif num_bits == 8:
        hqq_config = HqqConfig(
            nbits=8,
            group_size=hqq_base_config["q4_group_size"],
            quant_zero=hqq_base_config["quant_zero"],
            quant_scale=hqq_base_config["quant_scale"],
            axis=hqq_base_config["axis"]
        )
    elif num_bits == 4:
        hqq_config = HqqConfig(
            nbits=4,
            group_size=hqq_base_config["q4_group_size"],
            quant_zero=hqq_base_config["quant_zero"],
            quant_scale=hqq_base_config["quant_scale"],
            axis=hqq_base_config["axis"]
        )
    
    start_time = time.time()
    hqq_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=hqq_config, 
        torch_dtype=torch.float16,
        device_map=device,
        force_download=True
    )
    end_time = time.time()  # End time measurement
    hqq_model.QUANT_TIME = end_time - start_time

    dynamic_flag = "dynamic" if dynamic_config else ""
    hqq_model_name = f"{model_name.split('/')[1]}-hqq-{num_bits}bit-{dynamic_flag}"
    hqq_model_path = os.path.join(MODEL_SAVE_PATH, hqq_model_name)
    hqq_model.PATH = hqq_model_path
    hqq_model.NAME = hqq_model_name
    
    print(f'Model {model_name} is quantized')
    
    # Calculate model size and GPU utilization
    # calculate_model_size(bnb_model_path)
    # from src.models.utils_llm import print_gpu_utilization
    # print_gpu_utilization()

    return hqq_model
