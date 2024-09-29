import shutil
import re
import os
import subprocess
import tempfile

import logging

from src.evaluations.evaluate_reliability import evaluate_reliability

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
def run_evaluate(
    # Exp ID
    exp_id: str,
    save_excel: bool = True,
    num_excel_rows: int = 20,
    
    # Reliability dataset parameters
    seed=123,
    max_new_tokens=25,
    temperature=0.1,
    use_beam_search=False,
    strategy="Direct Completion",
    dataset_name="",
    typo_type="none",
    typo_intensity=0,
    n_repeats=10,
    n_beams=5,
    max_entries=None,
    
    # Model parameters
    seed_model=123,
    model_name="",
    model_path="",
    device="cuda",
    cache_path=CACHE_PATH
):
    ##################
    ## Print config ##
    ##################
    logger.info("Received the following configuration:")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Max new tokens: {max_new_tokens}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Use beam search: {use_beam_search}")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Dataset name: {dataset_name}")
    logger.info(f"  Number of repeats: {n_repeats}")
    logger.info(f"  Number of beams: {n_beams}")
    logger.info(f"  Max entries: {max_entries}")
    logger.info(f"  Seed model: {seed_model}")
    logger.info(f"  Model name: {model_name}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Device: {device}")

    results = evaluate_reliability(
        exp_id=exp_id,
        model_name=model_name,
        dataset_name=dataset_name,
        typo_type=typo_type,
        typo_intensity=typo_intensity,
        strategy=strategy,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_beam_search=use_beam_search,
        n_repeats=n_repeats,
        n_beams=n_beams,
        max_entries=max_entries,
        save_excel=save_excel,
        num_excel_rows=num_excel_rows,
        cache_dir=cache_path
    )

    fail_trace = {
        "fail_trace": seml.evaluation.get_results,
    }

    return {**results, **fail_trace}
