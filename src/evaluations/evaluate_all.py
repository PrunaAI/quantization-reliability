from typing import Dict
import logging

import torch
from src.evaluations.evaluate_brier_score import evaluate_brier_score
from src.evaluations.evaluate_memory import evaluate_model_size, evaluate_quantize_runtime, get_gpu_memory, record_gpu_memory
from src.evaluations.evaluate_perplexity import evaluate_perplexity

logger = logging.getLogger("quant_logger")

def evaluate(model, eval_dataloader, eval_metrics, device="cuda", to_device=False, prefix="", gpu_memory_usage={}) -> Dict:
    logger.info(f"Evaluating model with the following configuration:")
    logger.info(f"  Metrics: {eval_metrics}")
    logger.info(f"  Eval dataloader: {eval_dataloader.dataset.__class__.__name__}")
    
    model.eval()
    results = {}
    logger.info("Get device properties")
    if device == "cuda":
        results[f"{prefix}current_gpu_type"] = torch.cuda.get_device_properties(torch.cuda.device(0)).name
        results[f"{prefix}current_gpu_total_memory"] = (
            torch.cuda.get_device_properties(torch.cuda.device(0)).total_memory / 1024**2
        )
        results[f"{prefix}current_gpu_free_memory"] = get_gpu_memory()
        record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Evaluate GPU type")
        
    if "perplexity" in eval_metrics:
        logger.info("Evaluating Perplexity")
        results[f"{prefix}perplexity"] = evaluate_perplexity(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            to_device=to_device
        )
        record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Evaluate perplexity")
    if "brier_score" in eval_metrics:
        logger.info("Evaluating Brier Score")
        results[f"{prefix}brier_score"] = evaluate_brier_score(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            to_device=to_device
        )
        record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Evaluate brier score")
    if "model_size" in eval_metrics:
        logger.info("Evaluating Model Size")
        results[f"{prefix}model_size"] = evaluate_model_size(
            model=model
        )
        record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Evaluate model size")
    if "quantize_runtime" in eval_metrics:
        logger.info("Evaluating Quantize Runtime")
        results[f"{prefix}quantize_runtime"] = evaluate_quantize_runtime(
            model=model
        )
        record_gpu_memory(gpu_memory_usage=gpu_memory_usage, context="Evaluate quantize runtime")
    return results
