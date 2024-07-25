# LoRA configuration parameters
lora_params = {
    'r': 8,
    'lora_alpha': 32,
    'target_modules': ["q_proj", "k_proj", "o_proj"],
    'lora_dropout': 0.05,
    'bias': "none",
    'task_type': "CAUSAL_LM"
}

# Fine-tuning training hyperparameters
fine_tuning_params = {
    'grad_acc': 2,
    'logging_st': 1,
    'max_steps': -1,
    'lr': 1e-4,
    'batch_size': 1,
    'n_epochs': 1,
    'output_dir': '.',
    'fp16': True,
    'max_grad_norm': 1.0,
    'save_steps': 10000000,
    'lr_scheduler_type': "linear",
    'max_tokens': 256,
    'max_samples': 100,
    'random_seed': 100,
    'eval_dataset': None,
    'peft_config': None,
    'packing': True,
    'train_samples': 128
}

# AQLM_PREQUANTIZED configuration
AQLM_PREQUANTIZED = {
    "name": "AQLM-PREQUANTIZED",
    "quantize_method": "AQLM",
    "model_variant": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16",
    "model_variants": [
        "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16",
        "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16",
        "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16"
    ]
}

# AQLM_LORA configuration
AQLM_LORA = {
    "name": "AQLM-LORA",
    "quantize_method": "AQLM",
    "lora_params": lora_params,
    "fine_tuning_params": fine_tuning_params
}
