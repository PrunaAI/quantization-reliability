import shutil
import re
import os
import subprocess
import tempfile

import logging

from src.data.FKTC_datasets import load_dataset_from_name
from src.reliability.response_generator import ResponseGenerator
from src.models import base_models

logger = logging.getLogger("quant_logger")

logger.info("Setting up cache paths...")

# To avoid the following problem when running seml (see https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# Disables parallelism to remove transformers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CACHE_PATH = "/nfs/students/daro/.cache/huggingface"
HUB_PATH = os.path.join(CACHE_PATH, "hub")

if not os.path.exists(HUB_PATH):
    os.makedirs(HUB_PATH)
    print(f"Creating huggingface hub path at {HUB_PATH}")
    
print(f"Setting cache path to {CACHE_PATH}")
os.environ["TORCH_HOME"] = CACHE_PATH
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_PATH
os.environ["HUGGINGFACE_ASSETS_CACHE"] = CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = CACHE_PATH

import time
import random
import torch

import pandas as pd
import numpy as np
from sklearn import metrics

torch.hub.set_dir(CACHE_PATH)

# Empty the cache
import torch
with torch.no_grad():
    torch.cuda.empty_cache()
    
logging.info("Setting up working directory...")

#os.chdir('..')
logging.info(f"Current Working Directory: {os.getcwd()}")
import sys
sys.path.append("../") # Add directory containing src/data to path

import importlib
import src  # Assuming src is the package name

# Reload the src module after making changes
importlib.reload(src)

logging.info("Setting up working directory...")

# HuggingFace authentication
import os
from dotenv import load_dotenv
from huggingface_hub import login

logging.info("Authenticating Hugging Face...")

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
if huggingface_token is None:
    raise ValueError(
        f"Please set the HUGGINGFACE_TOKEN environment variable."
        f"Looking in {os.path.join(os.getcwd(), '.env')}"
    )
else:
    logging.info("Hugging Face token loaded successfully.")
login(token=huggingface_token, add_to_git_credential=True)

from seml.experiment import Experiment
import seml

logging.info("Setting up GPU memory usage list...")
# Global list to store GPU memory usage
gpu_memory_usage = {}

logging.info("Setting up SEML experiment...")

# Set the SEML experiment
ex = Experiment(save_git_info=False)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.automain
def run_reliability_eval(
    model_name,
    max_new_tokens,
    temperature,
    use_beam_search,
    strategy,
    dataset_name,
    taxonomy_type,
    device='cuda',
    seed=123,
    n_repeats=5,
    n_beams=5,
    cache_path=CACHE_PATH
    ):
    # LOAD DATASET
    qa_dataset = load_dataset_from_name(dataset_name, max_entries=None, taxonomy_type=taxonomy_type)

    # INITIALIZE RESULTS LIST
    results = []
    
    exp_id = "09-16-1"

    n_steps = 0
    total_steps = len(qa_dataset) * (n_repeats if not use_beam_search else 1)
    generator = ResponseGenerator(base_models[model_name])
    for query_idx, (query, true_answer) in enumerate(qa_dataset):
        run_results = generator.generate_response(query, strategy, dataset_name, true_answer, max_new_tokens, temperature, use_beam_search, n_repeats=n_repeats, n_beams=n_beams)
        for result_dict in run_results:
            print(f"  TOTAL: {n_steps + 1}/{total_steps}, MODEL: {model_name}, QUERY: {query_idx}, STRATEGY: {strategy}, MAX_NEW_TOKENS: {max_new_tokens}, RUN: {result_dict['run']}/{n_repeats}")
            # Store the results in the list
            results.append({
                "Query ID": query_idx,
                "Query": query,
                "Answer": true_answer,
                "Run": result_dict['run'],
                "Generated Response": result_dict['output_text'],
                "Cleaned": result_dict['cleaned'],
                "P": result_dict['beam_prob'],
                "P_adj": result_dict['beam_prob_adj'],
                "Entropy": result_dict['entropy'],
                "Is Correct": result_dict['is_correct'],
                "Token Probabilities": result_dict['token_probs']
            })
            print(f"    IS_CORRECT: {result_dict['is_correct']}, CLEANED: {result_dict['cleaned']}, PROB: {result_dict['beam_prob']:.2f}, ADJ_PROB: {result_dict['beam_prob_adj']:.2f}, ENTROPY: {result_dict['entropy']:.2f}")
            n_steps += 1
    
    # Generate custom file name based on parameters
    beam_search_str = "beam" if use_beam_search else "sample"
    strategy_str = strategy.replace(" ", "_").lower()  # Replace spaces with underscores for file names
    file_base = f"{model_name}_{dataset_name}_{taxonomy_type}_{beam_search_str}_{max_new_tokens}_tokens_{temperature}_temp_{strategy_str}"

    # Optional: Create a directory for saving the results if not already existing
    results_path = "/nfs/homedirs/daro/git/quantization-reliability/results"
    save_dir = os.path.join(results_path, "reliability_eval")
    os.makedirs(save_dir, exist_ok=True)

    # Generate file paths
    exp_path = os.path.join(save_dir, f"reliability_eval_{exp_id}")
    os.makedirs(exp_path, exist_ok=True)
    
    raw_table_path = os.path.join(exp_path, f"{file_base}_raw_table_{exp_id}.xlsx")
    scores_table_path = os.path.join(exp_path, f"{file_base}_scores_{exp_id}.xlsx")
    
    df_results = pd.DataFrame(results)
    
    # Calculate P_sem as the proportion of True values in 'Is Correct' per group
    df_results['P_sem'] = df_results.groupby(['Query ID'])['Is Correct'].transform('mean')

    # Define custom AUC calculation
    def custom_auc_roc(corrects, scores):
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    
    def calculate_scores(group):
        y_true = group['Is Correct'].values

        # Calculate various scores
        scores_dict = {}
        metrics_to_calculate = {
            'sample': 'P',
            'adj': 'P_adj',
            'sem': 'P_sem'
        }

        for key, score_column in metrics_to_calculate.items():
            y_scores = group[score_column].values
            if len(set(y_true)) > 1:  # Ensure at least two classes are present
                aucroc = custom_auc_roc(y_true, y_scores)
            else:
                aucroc = np.nan
            aucpr = metrics.average_precision_score(y_true, y_scores)
            accuracy = np.mean(y_true)
            brier_score = metrics.brier_score_loss(y_true, y_scores)
            log_loss = metrics.log_loss(y_true, y_scores)
            entropy = -np.sum(y_scores * np.log2(y_scores + 1e-10))  # Added small constant to avoid log(0)

            scores_dict[f'AUCROC_{key}'] = aucroc
            scores_dict[f'AUCPR_{key}'] = aucpr
            scores_dict[f'Brier_{key}'] = brier_score
            scores_dict[f'LogLoss_{key}'] = log_loss
            scores_dict[f'Entropy_{key}'] = entropy
            
        scores_dict[f'Accuracy'] = accuracy
        
        return pd.Series(scores_dict)

    # Apply the calculate_scores function to the entire DataFrame
    df_scores = calculate_scores(df_results)

    # Convert df_scores to a DataFrame with a single row for consistent saving format
    df_scores = df_scores.to_frame().T

    # Save the original detailed results to an Excel file
    df_results.to_excel(raw_table_path, index=False)
    df_scores.to_excel(scores_table_path, index=False)
    
    fail_trace = {
        "fail_trace": seml.evaluation.get_results,
    }

    return {"results": df_results, "scores": df_scores, **fail_trace}
