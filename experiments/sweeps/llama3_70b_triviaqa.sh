#!/bin/bash
# Llama-3 70B x TriviaQA  (replaces 03-15-llama3-70-all-triviaqa-a100-rerun)
# Requires 5x A100 — uses a100_multigpu hardware config

A100_MODELS="llama3_70b,llama3_70b_awq_4bit,\
llama3_70b_bnb_8bit,llama3_70b_bnb_4bit,\
llama3_70b_hqq_8bit,llama3_70b_hqq_4bit,llama3_70b_hqq_3bit,llama3_70b_hqq_2bit,llama3_70b_hqq_1bit,\
llama3_70b_quanto_8bit,llama3_70b_quanto_4bit,llama3_70b_quanto_2bit"

PERTURBATIONS="char_insertion,char_deletion,char_replacement,char_swapping,\
char_repetition,char_substitution,char_insert_noise,char_LCC,char_emoji,\
word_context_aware_insertion,word_keyword_only,word_swapping,word_repeat,\
word_internet_slang,word_phrase_translation"

cd "$(dirname "$0")/.."

# Base case
python run.py --multirun \
  exp_id="03-15-llama3-70-all-triviaqa-a100-rerun" \
  hardware=multi_gpu_5 \
  dataset=triviaqa \
  "model_name=$A100_MODELS"

# Perturbed case
python run.py --multirun \
  exp_id="03-15-llama3-70-all-triviaqa-a100-rerun-perturbed" \
  hardware=multi_gpu_5 \
  dataset=triviaqa_perturbed \
  "model_name=$A100_MODELS" \
  "dataset.perturbation_type=$PERTURBATIONS" \
  "dataset.perturbation_intensity=1,4,16"
