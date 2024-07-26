import os
import time
from awq import AutoAWQForCausalLM
from src import MODEL_SAVE_PATH

import logging
logger = logging.getLogger("quant_logger")

def quantize_awq(model_name, tokenizer, calib_dataloader, quantize_config={}, save_model=False, save_path="", device="cuda"):
    if quantize_config['num_bits'] is None or quantize_config['num_bits'] not in [4, 8]:
        raise ValueError(f"Invalid num_bits for AWQ: {quantize_config['num_bits']}")
        
    calib_text = []
    for i, batch in enumerate(calib_dataloader):
        if i >= quantize_config["n_samples"]:
            break
        input_ids, labels = batch
        decoded_text = tokenizer.decode(input_ids[0].tolist())
        calib_text.append(decoded_text)
        
    awq_config = {
        "zero_point": quantize_config["zero_point"],
        "q_group_size": quantize_config["q_group_size"],
        "w_bit": quantize_config["w_bit"],
        "version": quantize_config["version"]
    }

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
    logger.info(f"Quantization time: {awq_model.QUANT_TIME:.2f} seconds")

    awq_model_name = f"{model_name.split('/')[1]}-{quantize_config['name']}"
    awq_model_path = os.path.join(MODEL_SAVE_PATH, awq_model_name)
    os.makedirs(awq_model_path, exist_ok=True)
    awq_model.PATH = awq_model_path
    awq_model.NAME = awq_model_name
    
    print(f'Model {model_name} is quantized into {awq_model_name} and saved at "{awq_model_path}"')

    if save_model:
        save_dir = save_path if save_path else awq_model_path
        awq_model.save_quantized(save_dir)

    return awq_model
