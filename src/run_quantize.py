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
CACHE_PATH = "/nfs/students/daro/.cache/huggingface/"
print(f"Setting cache path to {CACHE_PATH}")

os.environ["TORCH_HOME"] = CACHE_PATH
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_PATH
os.environ["HUGGINGFACE_ASSETS_CACHE"] = CACHE_PATH

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
login(token=huggingface_token)

from seml.experiment import Experiment
import seml

from src.models import get_model, get_model_name
from src.data import data_loader_from_split, get_dataset
from src.algorithms.quantization.quantize import quantize
from src.evaluations.evaluate_all import evaluate
from src.evaluations.evaluate_memory import record_gpu_memory

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
def run_quantize(
    # Dataset parameters
    seed_dataset=123,
    directory_dataset="",
    calib_dataset_name="",
    calib_dataset_split="",
    eval_dataset_name="",
    eval_dataset_split="",
    batch_size=1,
    dataset_stride=1024,
    dataset_seq_length=1024,
    # Model parameters
    seed_model=123,
    directory_model="",
    clean_cache=True,
    model_name="",
    # Quantization parameters
    quantize_method="",
    num_bits=8,
    # Evaluation metrics
    eval_metrics=[
        "perplexity",
        "brier_score",
        # "model_size",
        # "gpu_utilization"
    ],
    device="cuda",
    save_quantized_model=False,
    quantized_model_save_path="",
):
    ##################
    ## Print config ##
    ##################
    logger.info("Received the following configuration:")
    logger.info(
        f"Calibration dataset: {calib_dataset_name}\n"
        f"Calibration split: {calib_dataset_split}\n"
        f"Evaluation dataset: {eval_dataset_name}\n"
        f"Evaluation split: {eval_dataset_split}\n"
        f"Dataloader stride: {dataset_stride}\n"
        f"Dataloader sequence length: {dataset_seq_length}\n"
        f"Batch size: {batch_size}\n"
        f"Model: {model_name}\n"
        f"Quantize method: {quantize_method}\n"
        f"Quantize bits: {num_bits}\n"
        f"Evaluation metrics: {eval_metrics}\n"
        f"Device: {device}\n"
    )
    
    ################
    ## Load model ##
    ################
    logger.info("Load base model")
    model_full_name = get_model_name(model_name)
    model, tokenizer = get_model(
        model_name=model_full_name,
        seed=seed_model,
        directory_model=directory_model,
        device=device,
    )
    record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Load model")
    
    ###############
    ## Load data ##
    ###############
    logger.info("Load calibration and evaluation data modules")
    calib_data_module = get_dataset(
        dataset_name=calib_dataset_name,
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=dataset_stride,
        tokenizer_name=model_full_name,
        seed=seed_dataset,
    )
    eval_data_module = get_dataset(
        dataset_name=eval_dataset_name,
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=dataset_stride,
        tokenizer_name=model_full_name,
        seed=seed_dataset,
    )
    
    calib_dataloader = data_loader_from_split(calib_data_module)[calib_dataset_split]
    eval_dataloader = data_loader_from_split(eval_data_module)[eval_dataset_split]
    record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Load data")

    ##############
    ## Quantize ##
    ##############
    logger.info("Quantization")
    quantized_model = quantize(
        model_name=model_full_name,
        tokenizer=tokenizer,
        calib_dataloader=calib_dataloader,
        quantize_method=quantize_method,
        num_bits=num_bits,
        save_model=save_quantized_model,
        save_path=quantized_model_save_path,
        device=device
    )
    record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Quantize model")

    ##############
    ## Evaluate ##
    ##############
    logger.info("Evaluating the quantized models")
    results = evaluate(
        model=quantized_model,
        eval_dataloader=eval_dataloader,
        eval_metrics=eval_metrics,
        factor=100,
        device=device,
        to_device=(quantize_method in ["AWQ"]),
        prefix="",
    )
    record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Evaluate model")
    
    ####################
    ## Cleaning cache ##
    ####################
    logger.info
    if clean_cache:
        for root, dirs, files in os.walk(CACHE_PATH, topdown=False):
            for dir_name in dirs:
                pattern = re.compile(f"^.*{model_full_name.split('/')[-1]}.*")
                dir_path = os.path.join(root, dir_name)
                if re.match(pattern, dir_path):
                    try:
                        shutil.rmtree(dir_path)
                    except:
                        pass

    fail_trace = {
        "fail_trace": seml.evaluation.get_results,
    }

    return {**results, **fail_trace}
