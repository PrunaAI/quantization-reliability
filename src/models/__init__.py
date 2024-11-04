import os
import torch
import logging
import re

from src import MODEL_SAVE_PATH
logger = logging.getLogger("quant_logger")

TINYLLAMA_CHAT = "TinyLlama-Chat"
TINYLLAMA = "TinyLlama"
LLAMA_3_8B = "Llama-3-8B"
BLOOMZ = "Bloomz"
GPT2_LARGE = "GPT2-Large"
LLAMA_3_8B_AQLM_2bit = "Llama-3-8B-AQLM-2bit"
LLAMA_3_8B_AQLM_PV_2bit = "Llama-3-8B-AQLM-PV-2bit"
LLAMA_3_8B_AQLM_PV_1bit = "Llama-3-8B-AQLM-PV-1bit"
LLAMA_3_8B_AWQ_4bit = "Llama-3-8B-AWQ-4bit"

LLAMA_3_8B_16K_BNB_4bit = "Llama-3-8B-16K-BNB-4bit"

LLAMA_3_8B_HQQ_4bit = "Llama-3-8B-HQQ-4bit"
LLAMA_3_8B_HQQ_2bit = "Llama-3-8B-HQQ-2bit"
LLAMA_3_8B_HQQ_1bit = "Llama-3-8B-HQQ-1bit"
LLAMA_3_8B_QUANTO_2bit = "Llama-3-8B-QUANTO-2bit"
LLAMA_3_8B_QUANTO_4bit = "Llama-3-8B-QUANTO-4bit"
LLAMA_3_8B_QUANTO_8bit = "Llama-3-8B-QUANTO-8bit"

LLAMA_3_8B_AWQ_4bit_local = "Llama-3-8B-AWQ-4bit-local"
LLAMA_3_8B_BNB_4bit_local = "Llama-3-8B-BNB-4bit-local"
LLAMA_3_8B_BNB_8bit_local = "Llama-3-8B-BNB-8bit-local"
LLAMA_3_8B_HQQ_8_uniform_local = "Llama-3-8B-HQQ-8-uniform-local"
LLAMA_3_8B_HQQ_mixed_local = "Llama-3-8B-HQQ-mixed-local"
LLAMA_3_8B_QUANTO_8_local = "Llama-3-8B-QUANTO-8-local"
LLAMA_3_8B_QUANTO_CALIB_8_local = "Llama-3-8B-QUANTO-CALIB-8-local"
LLAMA_3_8B_QUANTO_8_mixed_local = "Llama-3-8B-QUANTO-8-mixed-local"
LLAMA_3_8B_HQQ_LORA_local = "Llama-3-8B-HQQ-LORA-local"
LLAMA_3_8B_AQLM_LORA_local = "Llama-3-8B-AQLM-LORA-local"
META_LLAMA_3_8B = "Llama-3-8B"

# Base models dictionary
BASE_MODELS = {
    # TinyLlama models
    TINYLLAMA_CHAT: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    TINYLLAMA: "TinyLlama/TinyLlama_v1.1",
    
    # Meta models
    META_LLAMA_3_8B: "meta-llama/Meta-Llama-3-8B",
    
    # BigScience models
    BLOOMZ: "bigscience/bloomz-1b1",
    
    # OpenAI community models
    GPT2_LARGE: "openai-community/gpt2-large",
}

# Hugging Face quantized models dictionary
HF_QUANTIZED_MODELS = {
    # AQLM quantized models
    LLAMA_3_8B_AQLM_2bit: "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16",
    LLAMA_3_8B_AQLM_PV_2bit: "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16",
    LLAMA_3_8B_AQLM_PV_1bit: "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16",
    
    # AWQ quantized models
    LLAMA_3_8B_AWQ_4bit: "PrunaAI/meta-llama-Meta-Llama-3-8B-AWQ-4bit-smashed",
    
    # BitsAndBytes (BNB) quantized models
    LLAMA_3_8B_16K_BNB_4bit: "PrunaAI/mattshumer-Llama-3-8B-16K-bnb-4bit-smashed",
    
    # HQQ quantized models
    LLAMA_3_8B_HQQ_4bit: "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-4bit-smashed",
    LLAMA_3_8B_HQQ_2bit: "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-2bit-smashed",
    LLAMA_3_8B_HQQ_1bit: "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-1bit-smashed",
    
    # QUANTO quantized models
    LLAMA_3_8B_QUANTO_2bit: "PrunaAI/NousResearch-Meta-Llama-3-8B-QUANTO-int2bit-smashed",
    LLAMA_3_8B_QUANTO_4bit: "PrunaAI/NousResearch-Meta-Llama-3-8B-QUANTO-int4bit-smashed",
    LLAMA_3_8B_QUANTO_8bit: "PrunaAI/NousResearch-Meta-Llama-3-8B-QUANTO-int8bit-smashed",
}

LOCAL_QUANTIZED_MODELS = {
    # AWQ models
    LLAMA_3_8B_AWQ_4bit_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AWQ-4"),
    
    # BNB models
    LLAMA_3_8B_BNB_8bit_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-8"),
    LLAMA_3_8B_BNB_4bit_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-4"),
    
    # HQQ models
    LLAMA_3_8B_HQQ_8_uniform_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-8-uniform"),
    LLAMA_3_8B_HQQ_mixed_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-mixed"),
    
    # QUANTO models
    LLAMA_3_8B_QUANTO_8_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-8"),
    LLAMA_3_8B_QUANTO_CALIB_8_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-CALIB-8"),
    LLAMA_3_8B_QUANTO_8_mixed_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-8-mixed"),
    
    # HQQ-LORA models
    LLAMA_3_8B_HQQ_LORA_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-LORA"),
    
    # AQLM-LORA models
    LLAMA_3_8B_AQLM_LORA_local: os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AQLM-LORA"),
}

MODEL_TO_TOKENIZER_MAP = {
    LLAMA_3_8B_AWQ_4bit_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_BNB_4bit_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_BNB_8bit_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_HQQ_8_uniform_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_HQQ_mixed_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_QUANTO_8_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_QUANTO_CALIB_8_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_QUANTO_8_mixed_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_HQQ_LORA_local: BASE_MODELS[META_LLAMA_3_8B],
    LLAMA_3_8B_AQLM_LORA_local: BASE_MODELS[META_LLAMA_3_8B],
}

LLAMA_3_8B_MODEL_TO_CONFIG_MAP = {
    META_LLAMA_3_8B: "NONE",
    LLAMA_3_8B_BNB_4bit_local: "BNB-4",
    LLAMA_3_8B_BNB_8bit_local: "BNB-8",
    LLAMA_3_8B_AWQ_4bit_local: "AWQ-4",
    LLAMA_3_8B_HQQ_8_uniform_local: "HQQ-8-uniform",
    LLAMA_3_8B_HQQ_mixed_local: "HQQ-mixed",
    LLAMA_3_8B_HQQ_LORA_local: "HQQ-LORA",
    LLAMA_3_8B_QUANTO_8_local: "QUANTO",
    LLAMA_3_8B_QUANTO_CALIB_8_local: "QUANTO",
    LLAMA_3_8B_QUANTO_8_mixed_local: "QUANTO",
    LLAMA_3_8B_AQLM_LORA_local: "AQLM-LORA",
}

MODEL_NUM_BITS = {
    # Base models
    LLAMA_3_8B: 16,
    TINYLLAMA_CHAT: 16,
    TINYLLAMA: 16,
    BLOOMZ: 16,
    GPT2_LARGE: 32,
    
    # Local quantized models
    LLAMA_3_8B_BNB_8bit_local: 8,
    LLAMA_3_8B_HQQ_8_uniform_local: 8,
    LLAMA_3_8B_QUANTO_8_local: 8,
    LLAMA_3_8B_QUANTO_CALIB_8_local: 8,
    LLAMA_3_8B_QUANTO_8_mixed_local: 8,
    LLAMA_3_8B_BNB_4bit_local: 4,
    LLAMA_3_8B_AWQ_4bit_local: 4,
    LLAMA_3_8B_HQQ_mixed_local: 4,  # Using upper bound of 3-4 range
    
    # Hugging Face quantized models
    LLAMA_3_8B_QUANTO_8bit: 8,
    LLAMA_3_8B_HQQ_4bit: 4,
    LLAMA_3_8B_AWQ_4bit: 4,
    LLAMA_3_8B_16K_BNB_4bit: 4,
    LLAMA_3_8B_QUANTO_4bit: 4,
    LLAMA_3_8B_HQQ_2bit: 2,
    LLAMA_3_8B_AQLM_2bit: 2,
    LLAMA_3_8B_AQLM_PV_2bit: 2,
    LLAMA_3_8B_QUANTO_2bit: 2,
    LLAMA_3_8B_AQLM_PV_1bit: 1,
    LLAMA_3_8B_HQQ_1bit: 1
}