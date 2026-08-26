#!/bin/bash
# OPT KLD eval (replaces 26-03-21-opt-gtx, -a100, -70b-a100)

GTX_MODELS="opt_350m,opt_350m_bnb_8bit,opt_350m_bnb_4bit,\
opt_350m_gptq_8bit,opt_350m_gptq_4bit,opt_350m_gptq_3bit,opt_350m_gptq_2bit,\
opt_350m_hqq_8bit,opt_350m_hqq_4bit,opt_350m_hqq_3bit,opt_350m_hqq_2bit,\
opt_1.3b,opt_1.3b_bnb_8bit,opt_1.3b_bnb_4bit,\
opt_1.3b_gptq_8bit,opt_1.3b_gptq_4bit,opt_1.3b_gptq_3bit,opt_1.3b_gptq_2bit,\
opt_1.3b_hqq_8bit,opt_1.3b_hqq_4bit,opt_1.3b_hqq_3bit,opt_1.3b_hqq_2bit,\
opt_2.7b,opt_2.7b_bnb_8bit,opt_2.7b_bnb_4bit,\
opt_2.7b_hqq_8bit,opt_2.7b_hqq_4bit,opt_2.7b_hqq_3bit,opt_2.7b_hqq_2bit"

A100_MODELS="opt_2.7b_gptq_8bit,opt_2.7b_gptq_4bit,opt_2.7b_gptq_3bit,opt_2.7b_gptq_2bit,\
opt_6.7b,opt_6.7b_bnb_8bit,opt_6.7b_bnb_4bit,\
opt_6.7b_gptq_8bit,opt_6.7b_gptq_4bit,opt_6.7b_gptq_3bit,opt_6.7b_gptq_2bit,\
opt_6.7b_hqq_8bit,opt_6.7b_hqq_4bit,opt_6.7b_hqq_3bit,opt_6.7b_hqq_2bit,\
opt_13b,opt_13b_bnb_8bit,opt_13b_bnb_4bit,\
opt_13b_gptq_8bit,opt_13b_gptq_4bit,opt_13b_gptq_3bit,opt_13b_gptq_2bit,\
opt_13b_hqq_8bit,opt_13b_hqq_4bit,opt_13b_hqq_3bit,opt_13b_hqq_2bit"

cd "$(dirname "$0")/.."

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-opt-gtx" \
  hardware=single_gpu \
  "model_name=$GTX_MODELS"

python run_kld.py --multirun \
  exp_id="kl-divergence-eval-26-03-21-opt-a100" \
  hardware=single_gpu \
  "model_name=$A100_MODELS"
