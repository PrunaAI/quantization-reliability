#!/bin/bash
# Llama-3.2 3B x TriviaQA  (replaces 03-15-llama32-3-all-triviaqa-rerun + a100 variant)

GTX_MODELS="llama32_3b_hqq_4bit,llama32_3b_hqq_3bit,llama32_3b_hqq_2bit,llama32_3b_hqq_1bit,\
llama32_3b_quanto_4bit,llama32_3b_quanto_2bit,\
llama32_3b_bnb_4bit,\
llama32_3b_awq_4bit_local"

A100_MODELS="llama32_3b,llama32_3b_hqq_8bit,llama32_3b_quanto_8bit,llama32_3b_bnb_8bit,\
llama32_3b_gptq_8bit,llama32_3b_gptq_4bit,llama32_3b_gptq_3bit,llama32_3b_gptq_2bit,\
llama32_3b_aqlm_pv_2bit"

PERTURBATIONS="char_insertion,char_deletion,char_replacement,char_swapping,\
char_repetition,char_substitution,char_insert_noise,char_LCC,char_emoji,\
word_context_aware_insertion,word_keyword_only,word_swapping,word_repeat,\
word_internet_slang,word_phrase_translation"

cd "$(dirname "$0")/.."

# Base case — GTX
python run.py --multirun \
  exp_id="03-15-llama32-3-all-triviaqa-rerun" \
  hardware=single_gpu \
  dataset=triviaqa \
  "model_name=$GTX_MODELS"

# Base case — A100
python run.py --multirun \
  exp_id="03-15-llama32-3-all-triviaqa-a100-rerun" \
  hardware=single_gpu \
  dataset=triviaqa \
  "model_name=$A100_MODELS"

# Perturbed case — GTX
python run.py --multirun \
  exp_id="03-15-llama32-3-all-triviaqa-rerun-perturbed" \
  hardware=single_gpu \
  dataset=triviaqa_perturbed \
  "model_name=$GTX_MODELS" \
  "dataset.perturbation_type=$PERTURBATIONS" \
  "dataset.perturbation_intensity=1,4,16"
