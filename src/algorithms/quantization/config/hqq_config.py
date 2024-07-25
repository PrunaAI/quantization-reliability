import torch
from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig

##### HQQ 8-bit uniform

HQQ_8_uniform = {
    "name": "HQQ-8-uniform",
    "quantize_method": "HQQ",
    "num_bits": 8,
    "group_size": 64,
    "quant_zero": False,
    "quant_scale": False,
    "axis": 0
}

##### HQQ mixed

# Define 3-bit and 4-bit configurations separately
HQQ_q4_config = HQQBaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=False,
    quant_scale=False
)

HQQ_q3_config = HQQBaseQuantizeConfig(
    nbits=3,
    group_size=32,
    quant_zero=False,
    quant_scale=False
)

HQQ_mixed = {
    "name": "HQQ-mixed",
    "quantize_method": "HQQ",
    "num_bits": 4,
    "self_attn.q_proj": HQQ_q4_config,
    "self_attn.k_proj": HQQ_q4_config,
    "self_attn.v_proj": HQQ_q4_config,
    "self_attn.o_proj": HQQ_q4_config,
    "mlp.gate_proj": HQQ_q3_config,
    "mlp.up_proj": HQQ_q3_config,
    "mlp.down_proj": HQQ_q3_config
}

##### HQQ+

hqq_plus_base_params = {
    'nbits': 4,
    'group_size': 64,
    'quant_scale': False,
    'quant_zero': False
}

train_dtype = torch.float32
lora_base_params = {
    'lora_type': 'default',
    'r': 32,
    'lora_alpha': 64,
    'dropout': 0.05,
    'train_dtype': train_dtype
}

lora_params = {
    'self_attn.q_proj': lora_base_params,
    'self_attn.k_proj': lora_base_params,
    'self_attn.v_proj': lora_base_params,
    'self_attn.o_proj': lora_base_params,
    'mlp.gate_proj': None,
    'mlp.up_proj': None,
    'mlp.down_proj': None
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
    'fp16': (train_dtype == torch.float32),
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

HQQ_LORA = {
    "name": "HQQ-LORA",
    "quantize_method": "HQQ",
    "num_bits": 4,
    "base_params": hqq_plus_base_params,
    "lora_params": lora_params,
    "fine_tuning_params": fine_tuning_params
}
