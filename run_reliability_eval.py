import shutil
import re
import os
import subprocess
import tempfile
import logging

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
with torch.no_grad():
    torch.cuda.empty_cache()
    
print("Setting up working directory...")
print(f"Current Working Directory: {os.getcwd()}")
import sys
sys.path.append("../")

import os
import sys
import importlib.util

def troubleshoot_src_import():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    if os.path.isdir('src'):
        print("'src' directory found in current directory.")
        print("Contents of 'src' directory:")
        for item in os.listdir('src'):
            print(f"  {item}")
        
        if os.path.isfile(os.path.join('src', '__init__.py')):
            print("'src/__init__.py' found.")
        else:
            print("WARNING: 'src/__init__.py' not found. 'src' may not be a proper Python package.")
    else:
        print("'src' directory not found in current directory.")
        
        parent_dir = os.path.dirname(os.getcwd())
        if os.path.isdir(os.path.join(parent_dir, 'src')):
            print(f"'src' directory found in parent directory: {parent_dir}")
        else:
            print(f"'src' directory not found in parent directory: {parent_dir}")
    
    spec = importlib.util.find_spec("src")
    if spec is not None:
        print(f"'src' module found at: {spec.origin}")
    else:
        print("'src' module not found by importlib.")
    
    print("\nChecking all directories in sys.path:")
    for path in sys.path:
        if os.path.isdir(path):
            print(f"Directory: {path}")
            if 'src' in os.listdir(path):
                print(f"  'src' found in this directory")
                src_path = os.path.join(path, 'src')
                if os.path.isfile(os.path.join(src_path, '__init__.py')):
                    print(f"  'src' is a proper Python package (has __init__.py)")
                else:
                    print(f"  WARNING: 'src' is a directory but not a proper Python package (missing __init__.py)")

troubleshoot_src_import()

import importlib
import src
importlib.reload(src)

logging.info("Setting up working directory...")

# HuggingFace authentication
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

from src.evaluations.evaluate_reliability import evaluate_reliability

logging.info("Setting up GPU memory usage list...")
gpu_memory_usage = {}

logging.info("Setting up SEML experiment...")

ex = Experiment(save_git_info=False)

def validate_typo_config(typo_type, typo_intensity):
    """
    Validates the typo configuration based on the categories defined in the YAML.
    Returns True if the configuration is valid, False otherwise.
    """
    # Base case validation
    if typo_type == "none":
        return typo_intensity == 0
    
    # Specific perturbations validation
    if typo_type in ["word_remove_punctuation", "word_synonym"]:
        return typo_intensity == 1
    
    # Standard perturbations validation
    standard_perturbation_types = [
        "char_insertion", "char_deletion", "char_replacement",
        "char_repetition", "char_swapping", "word_CMW",
        "char_LCC", "char_insert_noise", "word_repeat",
        "char_substitution", "word_emoji", "word_internet_slang",
        "word_phrase_translation", "word_context_aware_insertion",
        "word_keyword_only"
    ]
    
    if typo_type in standard_perturbation_types:
        return typo_intensity in [1, 2, 3]
    
    return False

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.automain
def run_evaluate(
    # Exp ID
    exp_id: str,
    save_excel: bool = True,
    num_excel_rows: int = 20,
    batch_size: int = 32,
    
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
    max_relations=None,
    max_entries=None,
    
    # Model parameters
    seed_model=123,
    model_name="",
    model_path="",
    device="cuda",
    cache_path=CACHE_PATH
):
    # Validate typo configuration
    if not validate_typo_config(typo_type, typo_intensity):
        raise ValueError(
            f"Invalid typo configuration: type={typo_type}, intensity={typo_intensity}. "
            f"Please check the configuration categories in the YAML file."
        )

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
    logger.info(f"  Typo type: {typo_type}")
    logger.info(f"  Typo intensity: {typo_intensity}")
    logger.info(f"  Number of repeats: {n_repeats}")
    logger.info(f"  Number of beams: {n_beams}")
    logger.info(f"  Max relations: {max_relations}")
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
        max_relations=max_relations,
        max_entries=max_entries,
        save_excel=save_excel,
        num_excel_rows=num_excel_rows,
        cache_dir=cache_path,
        verbose=False,
        batch_size=batch_size
    )

    fail_trace = {
        "fail_trace": seml.evaluation.get_results,
    }

    return {**results, **fail_trace}