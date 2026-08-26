#!/bin/bash
# Llama-3 8B x TriviaQA  (replaces 03-15-llama3-all-triviaqa-a100-rerun)

A100_MODELS="llama3_8b,llama3_8b_awq_4bit_local,\
llama3_8b_bnb_8bit,llama3_8b_bnb_4bit,\
llama3_8b_gptq_8bit,llama3_8b_gptq_4bit,llama3_8b_gptq_3bit,llama3_8b_gptq_2bit,\
llama3_8b_hqq_8bit,llama3_8b_hqq_4bit,llama3_8b_hqq_3bit,llama3_8b_hqq_2bit,llama3_8b_hqq_1bit,\
llama3_8b_quanto_8bit,llama3_8b_quanto_4bit,llama3_8b_quanto_2bit,\
llama3_8b_quarot_8bit,llama3_8b_quarot_4bit,\
llama3_8b_qoq_4bit,llama3_8b_aqlm_2bit,llama3_8b_aqlm_pv_2bit,llama3_8b_aqlm_pv_1bit"

PERTURBATIONS="char_insertion,char_deletion,char_replacement,char_swapping,\
char_repetition,char_substitution,char_insert_noise,char_LCC,char_emoji,\
word_context_aware_insertion,word_keyword_only,word_swapping,word_repeat,\
word_internet_slang,word_phrase_translation"

cd "$(dirname "$0")/.."

# Base case
python run.py --multirun \
  exp_id="03-15-llama3-all-triviaqa-a100-rerun" \
  hardware=single_gpu \
  dataset=triviaqa \
  "model_name=$A100_MODELS"

# Perturbed case
python run.py --multirun \
  exp_id="03-15-llama3-all-triviaqa-a100-rerun-perturbed" \
  hardware=single_gpu \
  dataset=triviaqa_perturbed \
  "model_name=$A100_MODELS" \
  "dataset.perturbation_type=$PERTURBATIONS" \
  "dataset.perturbation_intensity=1,4,16"
