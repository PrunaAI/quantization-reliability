import os
import time
from awq import AutoAWQForCausalLM
from src import MODEL_SAVE_PATH

awq_base_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

def quantize_awq(model_name, tokenizer, calib_dataloader, num_bits=None, save_model=False, save_path="", device="cuda"):
    if num_bits is None or num_bits not in [4, 8]:
        raise ValueError(f"Invalid num_bits for AWQ: {num_bits}")

    awq_config = {}
    if num_bits == 8:
        awq_config = {
            "zero_point": awq_base_config["zero_point"],
            "q_group_size": awq_base_config["q_group_size"],
            "w_bit": 8,
            "version": awq_base_config["version"]
        }
    elif num_bits == 4:
        awq_config = {
            "zero_point": awq_base_config["zero_point"],
            "q_group_size": awq_base_config["q_group_size"],
            "w_bit": 4,
            "version": awq_base_config["version"]
        }
        
    calib_text = []
    for batch in calib_dataloader:
        input_ids, labels = batch
        decoded_text = tokenizer.decode(input_ids[0].tolist())
        calib_text.append(decoded_text)

    awq_model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        device_map=device
    )

    start_time = time.time()
    awq_model.quantize(
        tokenizer=tokenizer,
        quant_config=awq_config,
        calib_data=calib_text,
    )
    end_time = time.time()  # End time measurement
    awq_model.QUANT_TIME = end_time - start_time

    awq_model_name = f"{model_name.split('/')[1]}-awq-{num_bits}bit"
    awq_model_path = os.path.join(MODEL_SAVE_PATH, awq_model_name)
    os.makedirs(awq_model_path, exist_ok=True)
    awq_model.PATH = awq_model_path
    awq_model.NAME = awq_model_name
    
    print(f'Model is quantized and saved at "{awq_model_path}"')

    if save_model:
        save_dir = save_path if save_path else awq_model_path
        awq_model.save_quantized(save_dir)

    return awq_model
