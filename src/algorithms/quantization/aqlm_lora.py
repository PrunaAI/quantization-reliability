import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from src import MODEL_SAVE_PATH
from src.algorithms.quantization.config.aqlm_config import AQLM_MODEL_VARIANTS

import logging
logger = logging.getLogger("quant_logger")

def quantize_aqlm_lora(tokenizer, calib_dataloader, quantize_config={}, save_model=False, save_path="", device="cuda"):
    if quantize_config['num_bits'] is None or quantize_config['num_bits'] not in [1, 2]:
        raise ValueError(f"Invalid num_bits for BNB: {quantize_config['num_bits']}")
    
    if 'model_variant' not in quantize_config:
        raise ValueError("Missing required key 'model_variant' in quantize_config")
    
    # Validate the model_variant
    if quantize_config['model_variant'] not in AQLM_MODEL_VARIANTS:
        raise ValueError(f"Invalid model_variant: {quantize_config['model_variant']}")
    
    lora_params = quantize_config['lora_params']
    fine_tuning_params = quantize_config['fine_tuning_params']
    # Validate the LoRA configuration
    required_keys = ['r', 'lora_alpha', 'target_modules', 'lora_dropout', 'bias', 'task_type']
    for key in required_keys:
        if key not in lora_params:
            raise ValueError(f"Missing required LoRA config key: {key}")

    # Load the tokenizer and model
    model_name = quantize_config['model_variant']
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)
    model.NAME = model_name

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Configure and apply LoRA
    lora_params = LoraConfig(
        r=lora_params['r'],
        lora_alpha=lora_params['lora_alpha'],
        target_modules=lora_params['target_modules'],
        lora_dropout=lora_params['lora_dropout'],
        bias=lora_params['bias'],
        task_type=lora_params['task_type']
    )
    
    model = get_peft_model(model, lora_params)
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # Needed for gradient checkpointing

    # Load and preprocess dataset
    calib_dataset = calib_dataloader.ORIGINAL_DATASET
    calib_dataset = calib_dataset.select(range(fine_tuning_params['train_samples']))
    calib_dataset = calib_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)

    tokenizer.pad_token = tokenizer.eos_token

    # Define Trainer using fine-tuning parameters
    trainer = transformers.Trainer(
        model=model,
        train_dataset=calib_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=fine_tuning_params['batch_size'],
            gradient_accumulation_steps=fine_tuning_params['grad_acc'],
            gradient_checkpointing=True,
            warmup_steps=2,
            max_steps=fine_tuning_params['max_steps'],
            learning_rate=fine_tuning_params['lr'],
            fp16=fine_tuning_params['fp16'],
            logging_steps=fine_tuning_params['logging_st'],
            output_dir=fine_tuning_params['output_dir'],
            optim="adamw_bnb_8bit",
            logging_first_step=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!
    
    # Start finetuning
    start_time = time.time()
    trainer.train()
    end_time = time.time()  # End time measurement
    model.QUANT_TIME = end_time - start_time
    logger.info(f"Quantization and finetuning took {model.QUANT_TIME:.2f} seconds")

    model_name_suffix = f"{model_name.split('/')[1]}-LoRA"
    model_path = os.path.join(MODEL_SAVE_PATH, model_name_suffix)
    model.PATH = model_path
    model.NAME = model_name_suffix
    
    print(f'Model {model_name} is quantized and finetuned to {model_name_suffix}')

    # Save model if specified
    if save_model:
        save_dir = save_path if save_path else model_path
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    
    return model