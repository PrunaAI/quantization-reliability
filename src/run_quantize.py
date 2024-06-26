import shutil
import re
import os
import subprocess
import tempfile

# To avoid the following problem when running seml (see https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
if re.match(".*username.*", os.getcwd()):
    CACHE_PATH = "~/.cache/"
else:
    CACHE_PATH = "/tmp/"
os.environ["TORCH_HOME"] = CACHE_PATH
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_PATH
os.environ["HUGGINGFACE_ASSETS_CACHE"] = CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = CACHE_PATH

import time
import random
import torch

torch.hub.set_dir(CACHE_PATH)

import logging
logger = logging.getLogger("quant_logger")

#os.chdir('..')
print("Current Working Directory " , os.getcwd())
import sys
sys.path.append("../") # Add directory containing src/data to path

import importlib
import src  # Assuming src is the package name

# Reload the src module after making changes
importlib.reload(src)

from seml.experiment import Experiment
import seml

from src.models import get_model, get_model_name
from src.data import get_data_loader_from_split, get_dataset
from src.algorithms.quantization.quantize import quantize
from src.evaluations.evaluate_all import evaluate
from src.data import data_loader_map

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
    stride=512,
    # Model parameters
    seed_model=123,
    directory_model="",
    clean_cache=True,
    model_name="",
    # Quantization parameters
    quantize_method="",
    quantize_params={},
    # Evaluation metrics
    eval_metrics=[
        "perplexity",
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
        f"Batch size: {batch_size}\n"
        f"Model: {model_name}\n"
        f"Quantize method: {quantize_method}\n"
        f"Quantize params: {quantize_params}\n"
        f"Evaluation metrics: {eval_metrics}\n"
        f"Device: {device}\n"
    )
    
    ################
    ## Load model ##
    ################
    logger.info("Load base model")
    model_full_name = get_model_name[model_name]
    model, tokenizer = get_model(
        model_name=model_full_name,
        seed=seed_model,
        directory_model=directory_model,
        device=device,
    )
    
    ###############
    ## Load data ##
    ###############
    logger.info("Load calibration and evaluation data modules")
    calib_data_module = get_dataset(
        dataset_name=calib_dataset_name,
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=stride,
        tokenizer_name=model_full_name,
        seed=seed_dataset,
    )
    eval_data_module = get_dataset(
        dataset_name=eval_dataset_name,
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=stride,
        tokenizer_name=model_full_name,
        seed=seed_dataset,
    )
    
    calib_tokenizer = calib_data_module.tokenizer
    calib_dataloader = get_data_loader_from_split(calib_data_module, calib_dataset_split)
    
    eval_tokenizer = eval_data_module.tokenizer

    ################################
    ## Update quantize parameters ##
    ################################
    logger.info("Defining quantize parameters")
    quantize_params.update(quantize_params)
    logger.info(f"Default parameters adjusted from {quantize_params}")

    ##############
    ## Quantize ##
    ##############
    logger.info("Quantization")
    quantized_model, quantized_tokenizer = quantize(
        model=model,
        calib_tokenizer=calib_tokenizer,
        calib_dataloader=calib_dataloader,
        quantize_method=quantize_method,
        quantize_config=quantize_params,
        save_model=save_quantized_model,
        save_path=quantized_model_save_path,
        device=device
    )

    ##############
    ## Evaluate ##
    ##############
    logger.info("Evaluating the quantized models")
    results = evaluate(
        model=quantized_model,
        eval_tokenizer=eval_tokenizer,
        eval_data_module=eval_data_module,
        eval_metrics=eval_metrics,
        stride=stride,
        factor=100,
        device=device,
        prefix="",
    )

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
