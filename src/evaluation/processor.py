import time
import torch
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
import wandb
from src.loggers.wandb_utils import safe_wandb_log

from src.evaluation.metrics.brier import BrierScoreMetric
from src.evaluation.metrics.perplexity import PerplexityMetric
import logging


logger = logging.getLogger("reliability_eval")

class MetricProcessor:
    """Processes evaluation metrics for a model."""
    
    def __init__(self, device: str = "cuda"):
        # If device is 'auto', use cuda:0 for input tensors
        self.device = device
        self.metrics = {
            "perplexity": PerplexityMetric(device),
            "brier_score": BrierScoreMetric(device)
        }
    
    def process_batch(self, model: torch.nn.Module, batch: tuple, metric_names: List[str]) -> None:
        """Process a single batch through specified metrics with retry mechanism."""
        MAX_RETRIES = 3
        BASE_DELAY = 1  # Base delay in seconds
        logger.debug("Processing batch")
        
        inputs, targets = batch
        
        # If using a device_map, get the device of the first module parameter as a reference
        if self.device == 'auto':
            # Get the first device where model parameters are located
            param_device = next(model.parameters()).device
            inputs = inputs.to(param_device)
            targets = targets.to(param_device)
        else:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        
        for attempt in range(MAX_RETRIES):
            try:
                with torch.no_grad():
                    # Reset CUDA before processing attempt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Try with modifications if previous attempts failed
                    modified_inputs = inputs
                    
                    start_time = time.time()
                    outputs = model(modified_inputs)
                    processing_time = time.time() - start_time
                    safe_wandb_log({"model_processing_time_seconds": processing_time})
                    
                    logits = outputs.logits
                    logger.debug(f"Successfully processed batch")
                    logger.debug(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    
                    for metric_name in metric_names:
                        if metric_name in self.metrics:
                            self.metrics[metric_name].update(logits, targets)
                    
                    break  # If successful, exit the retry loop
                    
            except RuntimeError as e:
                if "CUDA" in str(e) or "an illegal memory access was encountered" in str(e):
                    # Device reset if CUDA error encountered
                    if torch.cuda.is_available():
                        try:
                            # More aggressive cleanup
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except:
                            pass
                    
                    # If this was our last attempt, raise the error
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed to process batch after {MAX_RETRIES} attempts: {str(e)}")
                        raise
                    
                    # Log retry attempt with more specific message for CUDA errors
                    delay = BASE_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"CUDA memory error on attempt {attempt + 1}: {str(e)}. Retrying in {delay} seconds with reduced memory usage...")
                    time.sleep(delay)
                else:
                    # For non-CUDA errors, handle as before
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # If this was our last attempt, raise the error
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed to process batch after {MAX_RETRIES} attempts: {str(e)}")
                        raise
                    
                    # Log retry attempt
                    delay = BASE_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Processing attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
    
    def compute_metrics(self, metric_names: List[str]) -> Dict[str, float]:
        """Compute final values for specified metrics."""
        return {
            name: self.metrics[name].compute()
            for name in metric_names
            if name in self.metrics
        }