import torch
import torchmetrics

from src.evaluation.metrics.base import BaseMetric

class PerplexityMetric(BaseMetric):
    """Computes perplexity for language model evaluation."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.metric = torchmetrics.text.Perplexity(ignore_index=-100).to(device)
        self.reset()
    
    def reset(self) -> None:
        """Reset perplexity metric."""
        self.metric.reset()
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update perplexity with batch predictions."""
        self.metric(logits.float(), targets)
    
    def compute(self) -> float:
        """Compute final perplexity score."""
        return self.metric.compute().item()
