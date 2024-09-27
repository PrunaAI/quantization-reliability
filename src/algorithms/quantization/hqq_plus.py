import os
import time
import torch
from transformers import TrainingArguments
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.models.hf.llama import LlamaHQQ
from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig
from hqq.core.peft import PeftUtils
from datasets import Dataset
from trl import SFTTrainer
from tqdm import tqdm
import numpy as np
import random
from src import MODEL_SAVE_PATH

import logging
logger = logging.getLogger("quant_logger")

def quantize_hqq_plus(model_name, tokenizer, calib_dataloader, quantize_config={}, save_model=False, device="cuda"):
    if quantize_config['num_bits'] is None or quantize_config['num_bits'] not in [4, 8]:
        raise ValueError(f"Invalid num_bits for HQQ: {quantize_config['num_bits']}")

    calib_dataset = calib_dataloader.ORIGINAL_DATASET
    hqq_config = {}
    if quantize_config['name'] == "HQQ-LORA":
        hqq_config = quantize_config
    else:
        raise ValueError(f"Invalid quantize_config for HQQ: {quantize_config['name']}")
    
    hqq_plus_base_params = hqq_config['base_params']
    lora_params = hqq_config['lora_params']
    fine_tuning_params = hqq_config['fine_tuning_params']    
    
    # Load model and tokenizer
    model = HQQModelForCausalLM.from_pretrained(model_name, cache_dir=None, torch_dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, cache_dir=None)

    # Quantize the model
    start_time = time.time()
    quant_config = HQQBaseQuantizeConfig(**hqq_plus_base_params)
    model.quantize_model(quant_config=quant_config)

    # Save model name and path
    hqq_model_name = f"{model_name.split('/')[1]}-{quantize_config['name']}.pt"
    hqq_model_path = os.path.join(MODEL_SAVE_PATH, hqq_model_name)
    model.NAME = hqq_model_name
    model.PATH = hqq_model_path

    # Add LoRA
    logger.info("Adding LoRA to model")
    PeftUtils.add_lora(model, lora_params)

    # Prepare dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    
    def pre_process_chat(chat):
        return chat

    random.seed(100)
    idx = random.sample(range(len(calib_dataset)), min(fine_tuning_params['max_samples'], fine_tuning_params['train_samples']))
    calib_dataset = Dataset.from_dict({'text': [pre_process_chat(calib_dataset[i]['text']) for i in tqdm(idx)]})
    # dataset_val = Dataset.from_dict({'text': [pre_process_chat(dataset_val[i]['text']) for i in range(len(dataset_val))]})

    # Train
    training_args = TrainingArguments(
        output_dir=fine_tuning_params['output_dir'],
        per_device_train_batch_size=fine_tuning_params['batch_size'],
        gradient_accumulation_steps=fine_tuning_params['grad_acc'],
        learning_rate=fine_tuning_params['lr'],
        logging_steps=fine_tuning_params['logging_st'],
        num_train_epochs=fine_tuning_params['n_epochs'],
        max_steps=fine_tuning_params['max_steps'],
        remove_unused_columns=False,
        fp16=fine_tuning_params['fp16'],
        max_grad_norm=fine_tuning_params['max_grad_norm'],
        save_steps=fine_tuning_params['save_steps'],
        lr_scheduler_type=fine_tuning_params['lr_scheduler_type'],
    )

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model.forward(*args, **kwargs)

        def train(self):
            self.model.train()

        def eval(self):
            self.model.eval()

        def parameters(self):
            return self.model.parameters()

    trainer = SFTTrainer(
        model=WrappedModel(model),
        tokenizer=tokenizer,
        max_seq_length=fine_tuning_params['max_tokens'],
        train_dataset=calib_dataset,
        eval_dataset=fine_tuning_params['eval_dataset'],
        peft_config=fine_tuning_params['peft_config'],
        packing=fine_tuning_params['packing'],
        args=training_args,
        dataset_text_field="text",
    )

    model.is_parallelizable = False
    trainer.is_model_parallel = False
    trainer.place_model_on_device = False

    model.train()
    try:
        trainer.train()
    except TypeError as e:
        logger.info(f"Run into error while saving model: {e}")
        
    end_time = time.time()
    model.QUANT_TIME = end_time - start_time
    logger.info(f"Quantization and finetuning took {model.QUANT_TIME:.2f} seconds")

    model.eval()
    logger.info("Casting LoRA weights to model dtype")
    PeftUtils.cast_lora_weights(model, dtype=torch.float32)
    
    # if save_model:
    #     save_dir = save_path if save_path else hqq_model_path
    #     PeftUtils.save_lora_weights(model, save_dir)
    #     PeftUtils.load_lora_weights(model, save_dir, base_class=LlamaHQQ)
    
    return model
