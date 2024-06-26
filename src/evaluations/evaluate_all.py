from typing import Dict
import logging

import torch
from src.evaluations.evaluate_brier_score import evaluate_brier_score
from src.evaluations.evaluate_memory import evaluate_gpu_utilization, evaluate_model_size
from src.evaluations.evaluate_perplexity import evaluate_perplexity

logger = logging.getLogger("quant_logger")

def evaluate(
    model,
    eval_tokenizer,
    eval_dataloader,
    eval_metrics,
    stride=512,
    factor=100,
    device="cuda",
    to_device=False,
    prefix="",
) -> Dict:
    """
    Evaluate the model with specified metrics.
    """
    model.eval()
    results = {}
    logger.info("Get device properties")
    if device == "cuda":
        results[f"{prefix}current_gpu_type"] = torch.cuda.get_device_properties(torch.cuda.device(0)).name
        results[f"{prefix}current_gpu_total_memory"] = (
            torch.cuda.get_device_properties(torch.cuda.device(0)).total_memory / 1024**2
        )
    if "perplexity" in eval_metrics:
        logger.info("Evaluate Perplexity")
        results[f"{prefix}perplexity"] = evaluate_perplexity(
            model=model,
            tokenizer=eval_tokenizer,
            dataloader=eval_dataloader,
            stride=stride,
            factor=factor,
            device=device,
            to_device=to_device
        )
    if "brier_score" in eval_metrics:
        logger.info("Evaluate Brier Score")
        results[f"{prefix}brier_score"] = evaluate_brier_score(
            model=model,
            tokenizer=eval_tokenizer,
            dataloader=eval_dataloader,
            stride=stride,
            factor=factor,
            device=device,
            to_device=to_device
        )
    if "model_size" in eval_metrics:
        logger.info("Evaluate Model Size")
        results[f"{prefix}model_size"] = evaluate_model_size(
            model_path=model.PATH
        )
    if "gpu_utilization" in eval_metrics:
        logger.info("Evaluate GPU Utilization")
        results[f"{prefix}gpu_utilization"] = evaluate_gpu_utilization()
    return results
