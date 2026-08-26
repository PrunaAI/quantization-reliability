from typing import Dict, List, Optional
import torch
import logging
from torch.utils.data import DataLoader

from src.evaluation.processor import MetricProcessor

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation across multiple metrics."""
    
    def __init__(self, device: str = "cuda"):
        # If device is 'auto', use cuda:0 for input tensors
        if device == 'auto':
            self.device = 'cuda:0'  # Use the first GPU for inputs
        else:
            self.device = device
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        metrics: List[str],
        n_samples: Optional[int] = None,
        to_device: bool = False,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Evaluate model on specified metrics."""
        logger.info(f"Starting evaluation with metrics: {metrics}")
        
        # Prepare model
        if to_device:
            model.to(self.device)
        model.eval()
        
        # Initialize results and processor
        results = self._get_device_info(prefix)
        processor = MetricProcessor(self.device)
        
        # Process batches
        for i, batch in enumerate(dataloader):
            if n_samples and i >= n_samples:
                break
            processor.process_batch(model, batch, metrics)
        
        # Compute metrics
        metric_results = processor.compute_metrics(metrics)
        results.update({f"{prefix}{k}": v for k, v in metric_results.items()})
        
        return results
    
    def _get_device_info(self, prefix: str) -> Dict[str, float]:
        """Get GPU device information if available."""
        results = {}
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(torch.cuda.device(0))
            results.update({
                f"{prefix}current_gpu_type": props.name,
                f"{prefix}current_gpu_total_memory": props.total_memory / 1024**2
            })
        return results
