from src.algorithms.quantization import QUANT_CONFIGS
from src.algorithms.quantization.awq import quantize_awq
from src.algorithms.quantization.bnb import quantize_bnb
from src.algorithms.quantization.hqq import quantize_hqq
from src.algorithms.quantization.hqq_plus import quantize_hqq_plus

import logging

from src.algorithms.quantization.quanto import quantize_quanto
logger = logging.getLogger("quant_logger")

# Wrapper class for quantization
def quantize(model_name, tokenizer, quantize_method, calib_dataloader=None, train_dataloader=None, save_model=False, save_path="", device="cuda"):
    logger.info(f"Quantizing model {model_name} with method {quantize_method} and calibration data {calib_dataloader.dataset.__class__.__name__}")
    model = None
    if quantize_method == "BNB-4":
        model = quantize_bnb(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS[quantize_method],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "BNB-8":
        model = quantize_bnb(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS[quantize_method],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "AWQ-4":
        model = quantize_awq(
            model_name=model_name,
            tokenizer=tokenizer,
            calib_dataloader=calib_dataloader,
            quantize_config=QUANT_CONFIGS[quantize_method],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "HQQ-8-uniform":
        model = quantize_hqq(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS[quantize_method],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "HQQ-mixed":
        model = quantize_hqq(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS[quantize_method],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "HQQ-LORA":
        model = quantize_hqq_plus(
            model_name=model_name,
            tokenizer=tokenizer,
            calib_dataloader=calib_dataloader,
            quantize_config=QUANT_CONFIGS[quantize_method],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "QUANTO" or quantize_method == "QUANTO-CALIB" or quantize_method == "QUANTO-QAT":
        model = quantize_quanto(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS[quantize_method],
            calib_dataloder=calib_dataloader,
            train_dataloader=train_dataloader,
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    else:
        raise NotImplementedError(f"Quantization method {quantize_method} should be one of {list(QUANT_CONFIGS.keys())}")

    return model
