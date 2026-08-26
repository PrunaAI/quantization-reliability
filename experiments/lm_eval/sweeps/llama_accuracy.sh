#!/bin/bash
# Accuracy eval (lm-eval harness) for LLaMA model family

GTX_MODELS="llama32_1b,llama32_1b_bnb_8bit,llama32_1b_bnb_4bit,\
llama32_1b_gptq_8bit,llama32_1b_gptq_4bit,llama32_1b_gptq_3bit,llama32_1b_gptq_2bit,\
llama32_1b_hqq_8bit,llama32_1b_hqq_4bit,llama32_1b_hqq_3bit,llama32_1b_hqq_2bit,llama32_1b_hqq_1bit,\
llama32_1b_quanto_8bit,llama32_1b_quanto_4bit,llama32_1b_quanto_2bit,llama32_1b_awq_4bit_local,\
llama32_3b,llama32_3b_bnb_8bit,llama32_3b_bnb_4bit,\
llama32_3b_gptq_8bit,llama32_3b_gptq_4bit,llama32_3b_gptq_3bit,llama32_3b_gptq_2bit,\
llama32_3b_hqq_8bit,llama32_3b_hqq_4bit,llama32_3b_hqq_3bit,llama32_3b_hqq_2bit,llama32_3b_hqq_1bit,\
llama32_3b_quanto_8bit,llama32_3b_quanto_4bit,llama32_3b_quanto_2bit,llama32_3b_awq_4bit_local"

A100_MODELS="llama3_8b,llama3_8b_bnb_8bit,llama3_8b_bnb_4bit,\
llama3_8b_gptq_8bit,llama3_8b_gptq_4bit,llama3_8b_gptq_3bit,llama3_8b_gptq_2bit,\
llama3_8b_hqq_8bit,llama3_8b_hqq_4bit,llama3_8b_hqq_3bit,llama3_8b_hqq_2bit,llama3_8b_hqq_1bit,\
llama3_8b_quanto_8bit,llama3_8b_quanto_4bit,llama3_8b_quanto_2bit,\
llama3_8b_awq_4bit_local,llama3_8b_quarot_8bit,llama3_8b_quarot_4bit,\
llama3_8b_qoq_4bit,llama3_8b_aqlm_2bit,llama3_8b_aqlm_pv_2bit"

A100_70B_MODELS="llama3_70b,llama3_70b_bnb_8bit,llama3_70b_bnb_4bit,\
llama3_70b_hqq_8bit,llama3_70b_hqq_4bit,llama3_70b_hqq_3bit,llama3_70b_hqq_2bit"

cd "$(dirname "$0")/.."

python run_lm_eval.py --multirun \
  exp_id="lm-eval-accuracy-llama-gtx" \
  hardware=single_gpu \
  "task=ceval,arc_easy,hellaswag,piqa,race" \
  "model_name=$GTX_MODELS"

python run_lm_eval.py --multirun \
  exp_id="lm-eval-accuracy-llama-a100" \
  hardware=single_gpu \
  "task=ceval,arc_easy,hellaswag,piqa,race" \
  "model_name=$A100_MODELS"

python run_lm_eval.py --multirun \
  exp_id="lm-eval-accuracy-llama-70b" \
  hardware=multi_gpu_5 \
  "task=ceval,arc_easy,hellaswag,piqa,race" \
  "model_name=$A100_70B_MODELS"
