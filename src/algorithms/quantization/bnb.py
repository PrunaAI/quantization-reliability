import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from accelerate import Accelerator
from src import MODEL_SAVE_PATH
from src.models.utils_llm import calculate_model_size

bnb_config = {
  "num_bits": 8,
  "llm_int8_threshold": 6.0,
  "llm_int8_enable_fp32_cpu_offload": False,
  "llm_int8_has_fp16_weight": False,
  "bnb_4bit_compute_dtype": torch.bfloat16,
  "bnb_4bit_quant_type": "fp4",
  "bnb_4bit_use_double_quant": False,
}

def quantize_bnb(model_name, quantize_config, save_model=False, save_path="", device="cuda"):
    if "num_bits" not in quantize_config or quantize_config["num_bits"] not in [4, 8]:
        raise ValueError(f"Invalid num_bits for BNB: {quantize_config.get('num_bits')}")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=(quantize_config["num_bits"] == 8),
        load_in_4bit=(quantize_config["num_bits"] == 4),
        llm_int8_threshold=quantize_config["llm_int8_threshold"] if quantize_config["num_bits"] == 8 else None,
        llm_int8_enable_fp32_cpu_offload=quantize_config["llm_int8_enable_fp32_cpu_offload"] if quantize_config["num_bits"] == 8 else None,
        llm_int8_has_fp16_weight=quantize_config["llm_int8_has_fp16_weight"] if quantize_config["num_bits"] == 8 else None,
        bnb_4bit_compute_dtype=quantize_config["bnb_4bit_compute_dtype"] if quantize_config["num_bits"] == 4 else None,
        bnb_4bit_quant_type=quantize_config["bnb_4bit_quant_type"] if quantize_config["num_bits"] == 4 else None,
        bnb_4bit_use_double_quant=quantize_config["bnb_4bit_use_double_quant"] if quantize_config["num_bits"] == 4 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    bnb_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        torch_dtype=torch.float32,
        device_map=device
    )

    model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name.split('/')[1]}-bnb-{quantize_config['num_bits']}bit")
    os.makedirs(model_path, exist_ok=True)
    accelerator = Accelerator()
    accelerator.save_model(bnb_model, model_path)
    
    # Calculate model size
    calculate_model_size(model_path)

    if save_model:
        save_dir = save_path if save_path else model_path
        bnb_model.save_pretrained(save_dir)

    return bnb_model, tokenizer
