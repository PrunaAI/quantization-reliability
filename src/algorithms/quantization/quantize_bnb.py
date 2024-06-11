import logging

from transformers import AutoModelForCausalLM
import torch


def quantize(model_name, bnb_config, device="cuda", logger_name="quant_logger"):
    """
    Quantizes a pre-trained model using BitsAndBytes configuration with logging.

    Args:
        model_name (str): Name of the pre-trained model to quantize.
        device (str): Device to load the model on (e.g., "cuda" or "cpu").
        bnb_config (BitsAndBytesConfig): Configuration for BitsAndBytes quantization.
        logger_name (str, optional): Name for the logger. Defaults to "quant_logger".

    Returns:
        AutoModelForCausalLM: The quantized model.
    """

    # Configure logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # Adjust logging level as needed

    # Create handler (optional, can redirect logs to a file)
    # handler = logging.FileHandler("quantization.log")
    # handler.setLevel(logging.INFO)
    # logger.addHandler(handler)  # Uncomment to add file logging

    try:
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float32,
            device_map=device
        )
        logger.info(f"Successfully loaded and quantized model: {model_name}")
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        raise e  # Re-raise the exception

    return quantized_model