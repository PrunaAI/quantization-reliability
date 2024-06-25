from typing import Dict
import logging

import torch
from src.evaluations.evaluate_brier_score import evaluate_brier_score
from src.evaluations.evaluate_perplexity import evaluate_perplexity

logger = logging.getLogger("quant_logger")


def evaluate(
    model,
    eval_tokenizer,
    eval_data_module,
    eval_metrics,
    stride=512,
    device="cuda",
    prefix="",
) -> Dict:
    """
    Evaluate the model with specified metrics.
    """
    results = {}
    logger.info("Get device properties")
    if device == "cuda":
        results[f"{prefix}current_gpu_type"] = torch.cuda.get_device_properties(torch.cuda.device(0)).name
        results[f"{prefix}current_gpu_total_memory"] = (
            torch.cuda.get_device_properties(torch.cuda.device(0)).total_memory / 1024**2
        )
        
    if "perplexity" in eval_metrics:
        logger.info("Evaluate perplexity")
        results[f"{prefix}perplexity"] = evaluate_perplexity(
            model=model,
            tokenizer=eval_tokenizer,
            data_module=eval_data_module,
            stride=stride,
            factor=100,
            device=device
        )
    if "brier_score" in eval_metrics:
        logger.info("Evaluate Brier score")
        results[f"{prefix}brier_score"] = evaluate_brier_score(
            model=model,
            tokenizer=eval_tokenizer,
            data_module=eval_data_module,
            stride=stride,
            factor=100,
            device=device
        )
    return results
