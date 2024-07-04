from src.algorithms.quantization.awq import quantize_awq, awq_config
from src.algorithms.quantization.bnb import quantize_bnb, bnb_config
from src.algorithms.quantization.hqq import quantize_hqq

# Wrapper class for quantization
def quantize(model_name, tokenizer, calib_dataloader, quantize_method, num_bits=None, save_model=False, save_path="", device="cuda"):        
    if quantize_method == "fp16":
        model = model
    elif quantize_method == "BNB":
        model = quantize_bnb(
            model_name=model_name,
            num_bits=num_bits,
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "AWQ":
        model = quantize_awq(
            model_name=model_name,
            tokenizer=tokenizer,
            calib_dataloader=calib_dataloader,
            num_bits=num_bits,
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "HQQ":
        model = quantize_hqq(
            model_name=model_name,
            num_bits=num_bits,
            dynamic_config=False,
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    else:
        raise NotImplementedError(f"Quantization method {quantize_method} not yet implemented.")

    return model
