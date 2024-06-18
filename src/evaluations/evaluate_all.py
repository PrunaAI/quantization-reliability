from typing import Dict
import logging

import torch
from src.evaluations.evaluate_text_generation import (
    evaluate_perplexity,
)

logger = logging.getLogger("quant_logger")


def evaluate(
    model,
    dataloader,
    evaluation_metrics,
    device="cuda",
    prefix="",
) -> Dict:
    """
    """
    results = {}
    logger.info("Get device properties")
    if device == "cuda":
        results[f"{prefix}current_gpu_type"] = torch.cuda.get_device_properties(torch.cuda.device(0)).name
        results[f"{prefix}current_gpu_total_memory"] = (
            torch.cuda.get_device_properties(torch.cuda.device(0)).total_memory / 1024**2
        )
    if "perplexity" in evaluation_metrics:
        logger.info("Evaluate perplexity")
        results[f"{prefix}perplexity"] = evaluate_perplexity(model=model, dataloader=dataloader, device=device)
    return results
