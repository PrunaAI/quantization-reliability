import os
import torch
import logging
import re


from src import MODEL_SAVE_PATH
from src.algorithms.quantization.config import QUANT_CONFIGS
logger = logging.getLogger("quant_logger")

# Base models dictionary
base_models = {
    # TinyLlama models
    "TinyLlama-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama": "TinyLlama/TinyLlama_v1.1",
    
    # Meta models
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    
    # BigScience models
    "Bloomz": "bigscience/bloomz-1b1",
    
    # OpenAI community models
    "GPT2-Large": "openai-community/gpt2-large",
}

# Hugging Face quantized models dictionary
hf_quantized_models = {
    # AQLM quantized models
    "Llama-3-8B-AQLM-2bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16",
    "Llama-3-8B-AQLM-PV-2bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16",
    "Llama-3-8B-AQLM-PV-1bit": "ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16",
    
    # AWQ quantized models
    "Llama-3-8B-AWQ-4bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-AWQ-4bit-smashed",
    
    # BitsAndBytes (BNB) quantized models
    "Llama-3-8B-16K-bnb-4bit": "PrunaAI/mattshumer-Llama-3-8B-16K-bnb-4bit-smashed",
    
    # HQQ quantized models
    "Llama-3-8B-HQQ-4bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-4bit-smashed",
    "Llama-3-8B-HQQ-2bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-2bit-smashed",
    "Llama-3-8B-HQQ-1bit": "PrunaAI/meta-llama-Meta-Llama-3-8B-HQQ-1bit-smashed",
    
    # QUANTO quantized models
    "Llama-3-8B-Quanto-2bit": "PrunaAI/NousResearch-Meta-Llama-3-8B-QUANTO-int2bit-smashed",
    "Llama-3-8B-Quanto-4bit": "PrunaAI/NousResearch-Meta-Llama-3-8B-QUANTO-int4bit-smashed",
    "Llama-3-8B-Quanto-8bit": "PrunaAI/NousResearch-Meta-Llama-3-8B-QUANTO-int8bit-smashed",
}

local_quantized_models = {
    # AWQ models
    "Llama-3-8B-AWQ-4bit-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AWQ-4"),
    
    # BNB models
    "Llama-3-8B-BNB-8bit-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-8"),
    "Llama-3-8B-BNB-4bit-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-BNB-4"),
    
    # HQQ models
    "Llama-3-8B-HQQ-8-uniform-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-8-uniform"),
    "Llama-3-8B-HQQ-mixed-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-mixed"),
    
    # QUANTO models
    "Llama-3-8B-QUANTO-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO"),
    "Llama-3-8B-QUANTO-CALIB-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-CALIB"),
    "Llama-3-8B-QUANTO-QAT-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-QUANTO-QAT"),
    
    # HQQ-LORA models
    "Llama-3-8B-HQQ-LORA-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-HQQ-LORA"),
    
    # AQLM-LORA models
    "Llama-3-8B-AQLM-LORA-local": os.path.join(MODEL_SAVE_PATH, "Meta-Llama-3-8B-AQLM-LORA"),
}

META_LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
local_tokenizers = {
    "Llama-3-8B-AWQ-4bit-local": META_LLAMA_3_8B,
    "Llama-3-8B-BNB-8bit-local": META_LLAMA_3_8B,
    "Llama-3-8B-BNB-4bit-local": META_LLAMA_3_8B,
    "Llama-3-8B-HQQ-8-uniform-local": META_LLAMA_3_8B,
    "Llama-3-8B-HQQ-mixed-local": META_LLAMA_3_8B,
    "Llama-3-8B-QUANTO-local": META_LLAMA_3_8B,
    "Llama-3-8B-QUANTO-CALIB-local": META_LLAMA_3_8B,
    "Llama-3-8B-QUANTO-QAT-local": META_LLAMA_3_8B,
    "Llama-3-8B-HQQ-LORA-local": META_LLAMA_3_8B,
    "Llama-3-8B-AQLM-LORA-local": META_LLAMA_3_8B,
}

LLAMA_3_8B_MODEL_TO_CONFIG_MAP = {
    "Llama-3-8B": "NONE",
    "Llama-3-8B-BNB-4bit-local": "BNB-4",
    "Llama-3-8B-BNB-8bit-local": "BNB-8",
    "Llama-3-8B-AWQ-4bit-local": "AWQ-4",
    "Llama-3-8B-HQQ-8-uniform-local": "HQQ-8-uniform",
    "Llama-3-8B-HQQ-mixed-local": "HQQ-mixed",
    "Llama-3-8B-HQQ-LORA-local": "HQQ-LORA",
    "Llama-3-8B-QUANTO-local": "QUANTO",
    "Llama-3-8B-QUANTO-CALIB-local": "QUANTO-CALIB",
    "Llama-3-8B-QUANTO-QAT-local": "QUANTO-QAT",
    "Llama-3-8B-AQLM-local": "AQLM",
    "Llama-3-8B-AQLM-LORA-local": "AQLM-LORA",
}

MODEL_NUM_BITS = {
    # Base models
    "Llama-3-8B": 16,
    "TinyLlama-Chat": 16,
    "TinyLlama": 16,
    "Bloomz": 16,
    "GPT2-Large": 32,
    
    # Local quantized models
    "Llama-3-8B-BNB-8bit-local": 8,
    "Llama-3-8B-HQQ-8-uniform-local": 8,
    "Llama-3-8B-BNB-4bit-local": 4,
    "Llama-3-8B-AWQ-4bit-local": 4,
    "Llama-3-8B-HQQ-mixed-local": 4,  # Using upper bound of 3-4 range
    
    # Hugging Face quantized models
    "Llama-3-8B-HQQ-4bit": 4,
    "Llama-3-8B-AWQ-4bit": 4,
    "Llama-3-8B-HQQ-2bit": 2,
    "Llama-3-8B-AQLM-2bit": 2,
    "Llama-3-8B-AQLM-PV-2bit": 2,
    "Llama-3-8B-AQLM-PV-1bit": 1,
    "Llama-3-8B-HQQ-1bit": 1
}