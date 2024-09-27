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

from src.models import get_model, get_model_name, get_tokenizer
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
    calib_dataset_name="",
    calib_dataset_split="",
    calib_seq_length=2048,
    batch_size=1,
    
    # Model parameters
    seed_model=123,
    directory_model="",
    model_name="Llama-3-8B",
    device="cuda",
    save_quantized_model=True,
    
    # Quantization parameters
    quantize_method="NONE",
):
    ##################
    ## Print config ##
    ##################
    logger.info("Received the following configuration:")
    logger.info(
        f"Calibration dataset: {calib_dataset_name}\n"
        f"Calibration split: {calib_dataset_split}\n"
        f"Calibration sequence length: {calib_seq_length}\n"
        f"Batch size: {batch_size}\n"
        f"Model: {model_name}\n"
        f"Quantize method: {quantize_method}\n"
        f"Device: {device}\n"
        f"Quantized model save: {save_quantized_model}\n"
    )
    
    with tempfile.TemporaryDirectory(prefix=CACHE_PATH) as temp_cache_dir:
        logger.info(f"Setting cache path to {temp_cache_dir}")
        os.environ["TORCH_HOME"] = temp_cache_dir
        os.environ["HF_HOME"] = temp_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = temp_cache_dir
        os.environ["HUGGINGFACE_ASSETS_CACHE"] = temp_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = temp_cache_dir
        torch.hub.set_dir(temp_cache_dir)
        
        ####################
        ## Load tokenizer ##
        ####################
        logger.info("Load tokenizer")
        model_full_name = get_model_name(model_name)
        tokenizer = get_tokenizer(
            model_name=model_full_name,
            seed=seed_model,
            directory_model=directory_model,
            device=device,
        )
        record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Load tokenizer")
        
        ###############
        ## Load data ##
        ###############
        logger.info("Load calibration data module")
        calib_data_module = get_dataset(
            dataset_name=calib_dataset_name,
            batch_size=batch_size,
            sequence_length=calib_seq_length,
            tokenizer_name=model_full_name,
            seed=seed_dataset,
        )
        
        logger.info("Load calibration dataloader")
        calib_dataloader = data_loader_from_split(
            data_module=calib_data_module,
            split=calib_dataset_split,
            sequence_length=512 if quantize_method in ["AWQ-4"] else calib_seq_length,
        )
        record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Load data")

        ##############
        ## Quantize ##
        ##############
        logger.info("Quantization...")
        quantized_model = None
        if quantize_method == "NONE":
            logger.info("Quantize method is None, loading original model")
            quantized_model = get_model(
                model_name=model_full_name,
                seed=seed_model,
                directory_model=directory_model,
                device=device,
            )
            record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Load base model")
        else:
            logger.info(f"Quantizing model using {quantize_method}")
            quantized_model = quantize(
                model_name=model_full_name,
                tokenizer=tokenizer,
                calib_dataloader=calib_dataloader,
                train_dataloader=calib_dataloader,
                quantize_method=quantize_method,
                save_model=save_quantized_model,
                device=device
            )
            record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Quantize model")

        ##############
        ## Evaluate ##
        ##############
        logger.info("Evaluation...")
        results = evaluate(
            model=quantized_model,
            eval_dataloader=eval_dataloader,
            eval_metrics=eval_metrics,
            n_samples=eval_n_samples,
            device=device,
            to_device=("AWQ" in quantize_method),
            prefix="",
            gpu_memory_usage=gpu_memory_usage
        )
        # record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Evaluate model")
        
        # ####################
        # ## Cleaning cache ##
        # ####################
        # logger.info("Cleaning cache...")
        # if clean_cache:
        #     for root, dirs, files in os.walk(HUB_PATH, topdown=False):
        #         for dir_name in dirs:
        #             pattern = re.compile(f"^.*{model_full_name.split('/')[-1]}.*")
        #             dir_path = os.path.join(root, dir_name)
        #             if re.match(pattern, dir_path):
        #                 try:
        #                     shutil.rmtree(dir_path)
        #                 except:
        #                     pass

    fail_trace = {
        "fail_trace": seml.evaluation.get_results,
    }

    return {**results, **fail_trace}
