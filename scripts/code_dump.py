import torch

dataset = [{"text": text} for text in wikitext_dataset]

samples = []
n_run = 0
n_samples=512
awq_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

for data in dataset:
    if isinstance(data, list):
        line_encoded = data
    else:
        line = data["text"]
        line = line.strip()
        line_encoded = awq_tokenizer.encode(line)
    if len(line_encoded) > 512:
        continue
    sample = torch.tensor([line_encoded])
    if sample.numel() == 0:
        continue
    samples.append(sample)
    n_run += 1
    if n_run == n_samples:
        break
# now concatenate all samples and split according to block size
print(samples)
cat_samples = torch.cat(samples, dim=1)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, AwqConfig
from accelerate.utils import load_and_quantize_model
from accelerate import init_empty_weights

device="cuda"

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # For instruction-based models.
# model_name = "meta-llama/Meta-Llama-3-8B"  # Too large to run on a gpu_gtx1080. GPU gpu_a100 is required.
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small enough to run on a gpu_gtx1080.
# model_name = "microsoft/Phi-3-vision-128k-instruct"  # Small enough to run on a gpu_gtx1080.

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    
# Setting quantization bits for the model
weight_quantization_bits = 8
double_quant = False

bnb_config = BitsAndBytesConfig(
    load_in_8bit=(weight_quantization_bits == 8),
    load_in_4bit=(weight_quantization_bits == 4),
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    # bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="fp4",
    # bnb_4bit_quant_type="nf4"
    bnb_4bit_use_double_quant=double_quant,
)

bnb_config.skip_modules = None

smashed_model_bnb = load_and_quantize_model(model, bnb_quantization_config=bnb_config, device_map=device)
print(smashed_model_bnb.__class__.__name__)

# Calibration Dataset needed - WikiText? Something else because data leakage? TODO: Explore
# smashed_model_awq = AutoModelForCausalLM.from_pretrained(
#     temp_dir, quantization_config=awq_config, trust_remote_code=True
# )

awq_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    do_fuse=False,
)

model.context_length()

import accelerate
print(accelerate.__file__)

import transformers
print(transformers.__file__)

from transformers import BitsAndBytesConfig
print(BitsAndBytesConfig.__dict__)

#!pip index versions accelerate
#!pip install accelerate --force-reinstall
!pip show transformers
!pip index versions transformers
!pip show accelerate
!pip index versions accelerate

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)