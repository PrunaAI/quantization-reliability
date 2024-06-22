import os
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from src import MODEL_SAVE_PATH
from src.models.utils_llm import calculate_model_size

awq_config = {
  "zero_point": True,
  "q_group_size": 128,
  "w_bit": 4,
  "version": "GEMM"
}

def quantize_awq(model_name, calib_tokenizer, calib_dataloader, quantize_config, save_model=False, save_path="", device="cuda"):
    if "num_bits" not in quantize_config or quantize_config["num_bits"] not in [4, 8]:
        raise ValueError(f"Invalid num_bits for AWQ: {quantize_config.get('num_bits')}")

    calib_text = []
    for batch in calib_dataloader:
        input_ids, labels = batch
        decoded_text = calib_tokenizer.decode(input_ids[0].tolist())
        calib_text.append(decoded_text)

    awq_config = {
        "zero_point": quantize_config.get("zero_point", True),
        "q_group_size": quantize_config.get("q_group_size", 128),
        "w_bit": quantize_config.get("w_bit", 4),
        "version": quantize_config.get("version", "GEMM")
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    awq_model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        device_map=device
    )

    awq_model.quantize(
        tokenizer=calib_tokenizer,
        quant_config=awq_config,
        calib_data=calib_text,
    )

    awq_model_name = f"{model_name.split('/')[1]}-awq"
    awq_model_path = os.path.join(MODEL_SAVE_PATH, awq_model_name)
    os.makedirs(awq_model_path, exist_ok=True)
    awq_model.save_quantized(awq_model_path)
    awq_model.tokenizer.save_pretrained(awq_model_path)
    awq_model.NAME = awq_model_name
    
    print(f'Model is quantized and saved at "{awq_model_path}"')
    
    # Calculate model size and GPU utilization
    calculate_model_size(awq_model_path)
    from src.models.utils_llm import print_gpu_utilization
    print_gpu_utilization()

    if save_model:
        save_dir = save_path if save_path else awq_model_path
        awq_model.save_pretrained(save_dir)

    return awq_model, tokenizer
