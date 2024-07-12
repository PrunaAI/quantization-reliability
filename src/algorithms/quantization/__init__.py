from src.algorithms.quantization.quantize import quantize
from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig

NONE = {
    "name": "NONE",
    "quantize_method": "NONE",
    "num_bits": 16
}

BNB_4 = {
    "name": "BNB-4",
    "quantize_method": "BNB",
    "num_bits": 4,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "fp4",
    "bnb_4bit_use_double_quant": False
}

BNB_8 = {
    "name": "BNB-8",
    "quantize_method": "BNB",
    "num_bits": 8,
    "load_in_8bit": True,
    "llm_int8_threshold": 6.0,
    "llm_int8_enable_fp32_cpu_offload": False,
    "llm_int8_has_fp16_weight": False
}

AWQ_4 = {
    "name": "AWQ-4",
    "quantize_method": "AWQ",
    "num_bits": 4,
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

HQQ_8_uniform = {
    "name": "HQQ-8-uniform",
    "quantize_method": "HQQ",
    "num_bits": 8,
    "group_size": 64,
    "quant_zero": False,
    "quant_scale": False,
    "axis": 0
}

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

QUANT_CONFIGS = {
  "NONE": NONE,
  "BNB-4": BNB_4,
  "BNB-8": BNB_8,
  "AWQ-4": AWQ_4,
  "HQQ-8-uniform": HQQ_8_uniform,
  "HQQ-mixed": HQQ_mixed
}
