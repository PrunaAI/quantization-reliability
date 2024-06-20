import logging
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from awq import AutoAWQForCausalLM
from accelerate import Accelerator
import os
from src.models.utils_llm import calculate_model_size

MODEL_SAVE_PATH = "./quantized_models"

def quantize(model_name, calib_dataloader, quantize_method, quantize_config, save_model=False, save_path=""):
    device = "cuda"

    if quantize_method == "BNB":
        if "num_bits" not in quantize_config or quantize_config["num_bits"] not in [4, 8]:
            raise ValueError(f"Invalid num_bits for BNB: {quantize_config.get('num_bits')}")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=(quantize_config["num_bits"] == 8),
            load_in_4bit=(quantize_config["num_bits"] == 4),
            llm_int8_threshold=6.0 if quantize_config["num_bits"] == 8 else None,
            llm_int8_enable_fp32_cpu_offload=False if quantize_config["num_bits"] == 8 else None,
            llm_int8_has_fp16_weight=False if quantize_config["num_bits"] == 8 else None,
            bnb_4bit_compute_dtype=torch.bfloat16 if quantize_config["num_bits"] == 4 else None,
            bnb_4bit_quant_type="fp4" if quantize_config["num_bits"] == 4 else None,
            bnb_4bit_use_double_quant=False if quantize_config["num_bits"] == 4 else None,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            torch_dtype=torch.float32,
            device_map=device
        )

        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name.split('/')[1]}-bnb-{quantize_config['num_bits']}bit")
        os.makedirs(model_path, exist_ok=True)
        accelerator = Accelerator()
        accelerator.save_model(model, model_path)
        calculate_model_size(model_path)

    elif quantize_method == "AWQ":
        if "num_bits" not in quantize_config or quantize_config["num_bits"] not in [4, 8]:
            raise ValueError(f"Invalid num_bits for AWQ: {quantize_config.get('num_bits')}")

        calib_text = []
        for batch in calib_dataloader:
            input_ids, labels = batch
            decoded_text = model.tokenizer.decode(input_ids[0].tolist())
            calib_text.append(decoded_text)

        awq_config = {
            "zero_point": quantize_config.get("zero_point", True),
            "q_group_size": quantize_config.get("q_group_size", 128),
            "w_bit": quantize_config.get("w_bit", 4),
            "version": quantize_config.get("version", "GEMM")
        }

        awq_model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            device_map=device
        )

        awq_model.quantize(
            tokenizer=model.tokenizer,
            quant_config=awq_config,
            calib_data=calib_text,
        )

        model = awq_model

        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name.split('/')[1]}-awq")
        os.makedirs(model_path, exist_ok=True)
        awq_model.save_quantized(model_path)
        awq_model.tokenizer.save_pretrained(model_path)
        calculate_model_size(model_path)

    else:
        raise NotImplementedError(f"Quantization method {quantize_method} not yet implemented.")

    if save_model:
        save_dir = save_path if save_path else model_path
        model.save_pretrained(save_dir)
        model.tokenizer.save_pretrained(save_dir)

    return model
