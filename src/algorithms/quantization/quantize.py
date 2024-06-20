import logging
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from awq import AutoAWQForCausalLM


def quantize(model, calib_dataloader, quantize_method, quantize_config, save_model=False, save_path=""):
    if quantize_method == "BNB":
        # Validate num_bits for BNB
        if "num_bits" not in quantize_config or quantize_config["num_bits"] not in [1, 2, 4, 8]:
            raise ValueError(f"Invalid num_bits for BNB: {quantize_config.get('num_bits')}")

        # Update quantize_config for BNB
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=(quantize_config["num_bits"] == 8),
            load_in_4bit=(quantize_config["num_bits"] == 4),
            **quantize_config  # Merge additional BNB configs
        )

        # Quantize model with BNB
        model.quantize(quantization_config=bnb_config)

    elif quantize_method == "AWQ":
        # Validate num_bits for AWQ
        if "num_bits" not in quantize_config or quantize_config["num_bits"] not in [1, 2, 4, 8]:
            raise ValueError(f"Invalid num_bits for AWQ: {quantize_config.get('num_bits')}")

        # Update quantize_config for AWQ
        awq_config = quantize_config.copy()  # Avoid modifying the original dict

        # Extract text from calibration dataloader
        calib_text = []
        for batch in calib_dataloader:
            # Assuming the batch contains input_ids (tokenized text) and labels (or None)
            input_ids, labels = batch
            decoded_text = model.tokenizer.decode(input_ids[0].tolist())  # Assuming first element in batch
            calib_text.append(decoded_text)

        # Quantize model with AWQ
        awq_model = AutoAWQForCausalLM.from_pretrained(model.config)
        awq_model.quantize(
            tokenizer=model.tokenizer,
            quant_config=awq_config,
            calib_data=calib_text,
        )

        model = awq_model  # Assign the quantized AWQ model to `model`

    else:
        raise NotImplementedError(f"Quantization method {quantize_method} not yet implemented.")

    if save_model:
        save_dir = save_path if save_path else "./quantized_model"
        model.save_pretrained(save_dir)
        model.tokenizer.save_pretrained(save_dir)

    return model

