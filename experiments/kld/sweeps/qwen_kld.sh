#!/bin/bash
# Qwen KLD eval (replaces 26-03-21-qwen-gtx, -a100, -14b-a100, -32b-a100)

A100_MODELS="qwen3_4b,qwen3_4b_bnb_8bit,qwen3_4b_bnb_4bit,\
qwen3_4b_gptq_8bit,qwen3_4b_gptq_4bit,qwen3_4b_gptq_3bit,qwen3_4b_gptq_2bit,\
qwen3_4b_hqq_8bit,qwen3_4b_hqq_4bit,qwen3_4b_hqq_3bit,qwen3_4b_hqq_2bit,\
qwen3_8b,qwen3_8b_bnb_8bit,qwen3_8b_bnb_4bit,\
qwen3_8b_gptq_8bit,qwen3_8b_gptq_4bit,qwen3_8b_gptq_3bit,qwen3_8b_gptq_2bit,\
qwen3_8b_hqq_8bit,qwen3_8b_hqq_4bit,qwen3_8b_hqq_3bit,qwen3_8b_hqq_2bit"

# 14B needs 4x A100 to load fp14b + quant14b simultaneously
A100_14B_MODELS="qwen3_14b,qwen3_14b_bnb_8bit,qwen3_14b_bnb_4bit,\
qwen3_14b_gptq_8bit,qwen3_14b_gptq_4bit,qwen3_14b_gptq_3bit,qwen3_14b_gptq_2bit,\
qwen3_14b_hqq_8bit,qwen3_14b_hqq_4bit,qwen3_14b_hqq_3bit,qwen3_14b_hqq_2bit"

# 32B needs 7x A100 to load fp32b + quant32b simultaneously
A100_32B_MODELS="qwen3_32b,qwen3_32b_bnb_8bit,qwen3_32b_bnb_4bit,\
qwen3_32b_gptq_8bit,qwen3_32b_gptq_4bit,qwen3_32b_gptq_3bit,qwen3_32b_gptq_2bit,\
qwen3_32b_hqq_8bit,qwen3_32b_hqq_4bit,qwen3_32b_hqq_3bit,qwen3_32b_hqq_2bit"

cd "$(dirname "$0")/.."

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-qwen-a100" \
  hardware=single_gpu \
  "model_name=$A100_MODELS"

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-qwen-14b-a100" \
  hardware=multi_gpu_4 \
  "model_name=$A100_14B_MODELS"

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-qwen-32b-a100" \
  hardware=multi_gpu_7 \
  "model_name=$A100_32B_MODELS"
