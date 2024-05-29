import shutil
import re
import os
import subprocess
import tempfile

# To avoid the following problem when running seml (see https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
if re.match(".*username.*", os.getcwd()):
    CACHE_PATH = "/ceph/hdd/staff/username/.cache/models"
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

from sacred import Experiment
import seml

from src.models import get_model
from src.data import get_dataset
from src.algorithms.quantization.quantize import quantize
from src.evaluations.evaluate_all import evaluate

# Set the SEML experiment
ex = Experiment(save_git_info=False)
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run_evaluate(
    # Dataset parameters,
    seed_dataset=123,
    directory_dataset="",
    calibration_dataset_name="",
    evaluation_dataset_name="",
    batch_size=1,
    # Model parameters
    seed_model=123,
    directory_model="",
    clean_cache=True,
    model_name="",
    weight_name="DEFAULT",
    task=None,
    # Evaluation metrics
    evaluation_metrics=[
        "perplexity",
    ],
    device="cuda",
):
    ###############
    ## Load data ##
    ###############
    print("Load datasets")
    calibration_data_module = get_dataset(
        dataset_name=calibration_dataset_name,
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        tokenizer_name=model_name,
        seed=seed_dataset,
    )
    calibration_dataloader = calibration_data_module.val_dataloader()
    evaluation_data_module = get_dataset(
        dataset_name=evaluation_dataset_name,
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        tokenizer_name=model_name,
        seed=seed_dataset,
    )
    evaluation_dataloader = evaluation_data_module.test_dataloader()

    model = get_model(
        model_name=model_name,
        weight_name=weight_name,
        task=task,
        seed=seed_model,
        directory_model=directory_model,
    )

    # TODO: quantize the model
    model = quantize(model, calibration_dataloader)

    results = evaluate(
        model=model,
        dataloader=evaluation_dataloader,
        evaluation_metrics=evaluation_metrics,
        device=device,
        prefix="",
    )

    if clean_cache:
        for root, dirs, files in os.walk(CACHE_PATH, topdown=False):
            for dir_name in dirs:
                pattern = re.compile(f"^.*{model_name.split('/')[-1]}.*")
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
