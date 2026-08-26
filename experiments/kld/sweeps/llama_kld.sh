#!/bin/bash
# Llama KLD eval (replaces 26-03-21-llama-gtx, -a100, -70b-a100)

GTX_MODELS="llama32_1b,llama32_1b_bnb_8bit,llama32_1b_bnb_4bit,\
llama32_1b_gptq_8bit,llama32_1b_gptq_4bit,llama32_1b_gptq_3bit,llama32_1b_gptq_2bit,\
llama32_1b_hqq_8bit,llama32_1b_hqq_4bit,llama32_1b_hqq_3bit,llama32_1b_hqq_2bit,llama32_1b_hqq_1bit,\
llama32_1b_quanto_8bit,llama32_1b_quanto_4bit,llama32_1b_quanto_2bit,\
llama32_1b_awq_4bit_local,\
llama32_3b_hqq_4bit,llama32_3b_hqq_3bit,llama32_3b_hqq_2bit,llama32_3b_hqq_1bit,\
llama32_3b_quanto_4bit,llama32_3b_quanto_2bit,\
llama32_3b_bnb_4bit,llama32_3b_awq_4bit_local"

A100_MODELS="llama32_3b,llama32_3b_hqq_8bit,\
llama32_3b_gptq_8bit,llama32_3b_gptq_4bit,llama32_3b_gptq_3bit,llama32_3b_gptq_2bit,\
llama32_3b_bnb_8bit,llama32_3b_quanto_8bit,llama32_3b_quanto_4bit,llama32_3b_quanto_2bit,\
llama32_3b_awq_4bit_local,\
llama3_8b,llama3_8b_hqq_8bit,llama3_8b_hqq_4bit,llama3_8b_hqq_3bit,llama3_8b_hqq_2bit,\
llama3_8b_bnb_8bit,llama3_8b_bnb_4bit,\
llama3_8b_quanto_8bit,llama3_8b_quanto_4bit,llama3_8b_quanto_2bit,\
llama3_8b_awq_4bit_local"

# 8B GPTQ + 70B models need 7 GPUs to load fp+quant simultaneously
A100_7GPU_MODELS="llama3_8b_gptq_8bit,llama3_8b_gptq_4bit,llama3_8b_gptq_3bit,llama3_8b_gptq_2bit,\
llama3_70b,llama3_70b_awq_4bit,llama3_70b_bnb_8bit,llama3_70b_bnb_4bit,\
llama3_70b_hqq_8bit,llama3_70b_hqq_4bit,llama3_70b_hqq_3bit,llama3_70b_hqq_2bit,llama3_70b_hqq_1bit,\
llama3_70b_quanto_8bit,llama3_70b_quanto_4bit,llama3_70b_quanto_2bit"

cd "$(dirname "$0")/.."

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-llama-gtx" \
  hardware=single_gpu \
  "model_name=$GTX_MODELS"

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-llama-a100" \
  hardware=single_gpu \
  "model_name=$A100_MODELS"

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-llama-70b-a100" \
  hardware=multi_gpu_7_high \
  "model_name=$A100_7GPU_MODELS"
