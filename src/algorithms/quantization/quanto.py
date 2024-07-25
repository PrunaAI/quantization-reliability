import os
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.quanto import Calibration, quantize, freeze, qint8, qint4, safe_save, QTensor
from torch.optim import Adam
from src import MODEL_SAVE_PATH

import logging
logger = logging.getLogger("quant_logger")

keyword_to_itype = {
    "qint8": qint8,
    "qint4": qint4
}

def quantize_quanto(model_name, quantize_config={}, calib_dataloder=None, train_dataloader=None, save_model=False, save_path="", device="cuda"):
    if 'name' not in quantize_config:
        raise ValueError("Invalid quantize_config: 'name' key is missing")

    start_time = time.time()

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    weights = keyword_to_itype[quantize_config.get("weights", "qint8")] 
    activations = keyword_to_itype[quantize_config.get("activations", "qint8")]

    # Perform quantization based on the config
    if quantize_config['name'] == "QUANTO":
        logger.info("Performing quantization for QUANTO")
        quantize(model, weights=weights, activations=activations)

    elif quantize_config['name'] == "QUANTO-CALIB":  # Too memory-intensive for Llama-3-8B
        logger.info("Performing calibration for QUANTO")
        quantize(model, weights=weights, activations=activations)
        cal_dataset = calib_dataloder.ORIGINAL_DATASET
        cal_samples = quantize_config.get("n_samples", 128)
        with Calibration(streamline=True, debug=False):
            model.eval()
            total = 0
            for batch in cal_dataset.iter(batch_size=1):
                inputs = tokenizer(batch["text"], return_tensors="pt", padding=True).to(device)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                model(input_ids, attention_mask=attention_mask)
                total += input_ids.size(0)
                if total >= cal_samples:
                    break

    elif quantize_config['name'] == "QUANTO-QAT":  # Too memory-intensive for Llama-3-8B
        logger.info("Performing Quantization Aware Training for QUANTO")
        quantize(model, weights=weights, activations=activations)
        train_samples = quantize_config.get("train_samples", 128)
        lr = quantize_config.get("lr", 1e-4)
        model.to(device)
        model.train()
        optimizer = Adam(model.parameters(), lr=lr)
        n_epochs = 5  # Set to lower values for debugging
        for epoch in range(n_epochs):
            for batch_idx, (data, target) in tqdm(enumerate(train_dataloader)):
                print(f"Epoch: {epoch}/{n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}")
                if batch_idx >= train_samples:
                    break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(data)
                    logits = output.logits
                    if isinstance(logits, QTensor):
                        logits = logits.dequantize()
                    loss = torch.nn.functional.nll_loss(logits.view(1, logits.shape[2], logits.shape[1]), target)
                    loss.backward()
                    optimizer.step()

    else:
        raise ValueError(f"Invalid quantize_config for QUANTO: {quantize_config['name']}")

    # Freeze the model to convert weights to integers
    freeze(model)

    end_time = time.time()
    model.QUANT_TIME = end_time - start_time

    model_name_suffix = quantize_config['name']
    model_save_name = f"{model_name.split('/')[1]}-{model_name_suffix}"
    model_save_path = os.path.join(MODEL_SAVE_PATH, model_save_name)
    model.PATH = model_save_path
    model.NAME = model_save_name

    print(f'Model {model_name} is quantized to {model_save_name}')

    if save_model:
        save_dir = save_path if save_path else model_save_path
        safe_save(model.state_dict(), f"{save_dir}.safetensors")

    return model