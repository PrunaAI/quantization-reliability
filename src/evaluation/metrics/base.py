from abc import ABC, abstractmethod
import torch

class BaseMetric(ABC):
    """Base class for all evaluation metrics."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update metric with new batch."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute final metric value."""
        pass