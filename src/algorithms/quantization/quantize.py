from src.algorithms.quantization import QUANT_CONFIGS
from src.algorithms.quantization.awq import quantize_awq
from src.algorithms.quantization.bnb import quantize_bnb
from src.algorithms.quantization.hqq import quantize_hqq

# Wrapper class for quantization
def quantize(model_name, tokenizer, calib_dataloader, quantize_method, num_bits=None, save_model=False, save_path="", device="cuda"):        
    if quantize_method == "BNB-4":
        model = quantize_bnb(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS["BNB_4"],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "BNB-8":
        model = quantize_bnb(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS["BNB_8"],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "AWQ-4":
        model = quantize_awq(
            model_name=model_name,
            tokenizer=tokenizer,
            calib_dataloader=calib_dataloader,
            quantize_config=QUANT_CONFIGS["AWQ_4"],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "HQQ-8-uniform":
        model = quantize_hqq(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS["HQQ_8_uniform"],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "HQQ-mixed":
        model = quantize_hqq(
            model_name=model_name,
            quantize_config=QUANT_CONFIGS["HQQ_mixed"],
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    else:
        raise NotImplementedError(f"Quantization method {quantize_method} not yet implemented.")

    return model
