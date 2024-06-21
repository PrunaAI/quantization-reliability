from awq import quantize_awq, awq_config
from bnb import quantize_bnb, bnb_config

quantize_config_map = {
    "BNB": bnb_config,
    "AWQ": awq_config,
}

# Wrapper class for quantization
def quantize(model_name, calib_tokenizer, calib_dataloader, quantize_method, quantize_config=None, save_model=False, save_path="", device="cuda"):    
    if quantize_config is None:
        quantize_config = quantize_config_map[quantize_method]
        
    if quantize_method == "BNB":
        model, tokenizer = quantize_bnb(
            model_name=model_name,
            quantize_config=quantize_config,
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    elif quantize_method == "AWQ":
        model, tokenizer = quantize_awq(
            model_name=model_name,
            calib_tokenizer=calib_tokenizer,
            calib_dataloader=calib_dataloader,
            quantize_config=quantize_config,
            save_model=save_model,
            save_path=save_path,
            device=device
        )
    else:
        raise NotImplementedError(f"Quantization method {quantize_method} not yet implemented.")

    return model, tokenizer
