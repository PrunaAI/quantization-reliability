import torch
from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig

##### NONE

NONE = {
    "name": "NONE",
    "quantize_method": "NONE",
    "num_bits": 16
}

##### BNB 4-bit

BNB_4 = {
    "name": "BNB-4",
    "quantize_method": "BNB",
    "num_bits": 4,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "fp4",
    "bnb_4bit_use_double_quant": False
}

##### BNB 8-bit

BNB_8 = {
    "name": "BNB-8",
    "quantize_method": "BNB",
    "num_bits": 8,
    "load_in_8bit": True,
    "llm_int8_threshold": 6.0,
    "llm_int8_enable_fp32_cpu_offload": False,
    "llm_int8_has_fp16_weight": False
}

##### AWQ 4-bit

AWQ_4 = {
    "name": "AWQ-4",
    "quantize_method": "AWQ",
    "num_bits": 4,
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
    "n_samples": 128
}

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

QUANTO = {
    "name": "QUANTO",
    "quantize_method": "QUANTO",
    "num_bits": 8,
    "weights": "qint8",
    "activations": "qint8",
}

QUANTO_CALIB = {
    "name": "QUANTO-CALIB",
    "quantize_method": "QUANTO",
    "num_bits": 8,
    "weights": "qint8",
    "activations": "qint8",
    "n_samples": 128,
    "momentum": 0.9,
}

QUANTO_QAT = {
    "name": "QUANTO-QAT",
    "quantize_method": "QUANTO",
    "num_bits": 8,
    "weights": "qint8",
    "activations": "qint8",
    "train_samples": 128,
    "lr": 1e-4,
}

QUANT_CONFIGS = {
    "NONE": NONE,
    "BNB-4": BNB_4,
    "BNB-8": BNB_8,
    "AWQ-4": AWQ_4,
    "HQQ-8-uniform": HQQ_8_uniform,
    "HQQ-mixed": HQQ_mixed,
    "HQQ-LORA": HQQ_LORA,
    "QUANTO": QUANTO,
    "QUANTO-CALIB": QUANTO_CALIB,
    "QUANTO-QAT": QUANTO_QAT
}
