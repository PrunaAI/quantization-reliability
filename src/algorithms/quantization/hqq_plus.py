import os
import time
import torch
from transformers import AutoModelForCausalLM
from hqq.models.hf.base import AutoHQQHFModel
from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig
from src import MODEL_SAVE_PATH

def quantize_hqq_plus(model_name, quantize_config={}, save_model=False, save_path="", device="cuda"):
    if quantize_config['num_bits'] is None or quantize_config['num_bits'] not in [4, 8]:
        raise ValueError(f"Invalid num_bits for HQQ: {quantize_config['num_bits']}")

    hqq_config = {}
    # Option 1: All linear layers will use the same quantization config
    if quantize_config['name'] == "HQQ-mixed":
        hqq_config = {
            'self_attn.q_proj': quantize_config['self_attn.q_proj'],
            'self_attn.k_proj': quantize_config['self_attn.k_proj'],
            'self_attn.v_proj': quantize_config['self_attn.v_proj'],
            'self_attn.o_proj': quantize_config['self_attn.o_proj'],
            'mlp.gate_proj': quantize_config['mlp.gate_proj'],
            'mlp.up_proj': quantize_config['mlp.up_proj'],
            'mlp.down_proj': quantize_config['mlp.down_proj'],
        }
    elif quantize_config['name'] == "HQQ-8-uniform":
        hqq_config = HQQBaseQuantizeConfig(
            nbits=quantize_config['num_bits'],
            group_size=quantize_config['group_size'],
            quant_zero=quantize_config["quant_zero"],
            quant_scale=quantize_config["quant_scale"],
            axis=quantize_config["axis"]
        )
    else:
        raise ValueError(f"Invalid quantize_config for HQQ: {quantize_config['name']}")
    
    start_time = time.time()
    hqq_model_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")
    hqq_model = AutoHQQHFModel.quantize_model(
        hqq_model_base,
        quant_config=hqq_config,
        compute_dtype=torch.float16,
        device=device
    )
    end_time = time.time()  # End time measurement
    hqq_model.QUANT_TIME = end_time - start_time

    hqq_model_name = f"{model_name.split('/')[1]}-{quantize_config['name']}"
    hqq_model_path = os.path.join(MODEL_SAVE_PATH, hqq_model_name)
    hqq_model.PATH = hqq_model_path
    hqq_model.NAME = hqq_model_name
    
    print(f'Model {model_name} is quantized to {hqq_model_name}')
    
    # Calculate model size and GPU utilization
    # calculate_model_size(bnb_model_path)
    # from src.models.utils_llm import print_gpu_utilization
    # print_gpu_utilization()
    
    if save_model:
        save_dir = save_path if save_path else hqq_model_path
        AutoHQQHFModel.save_quantized(hqq_model, save_dir)

    return hqq_model
