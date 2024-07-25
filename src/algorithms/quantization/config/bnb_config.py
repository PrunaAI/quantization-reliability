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