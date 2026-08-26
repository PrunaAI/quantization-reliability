import torch
import torch.nn.functional as F
from src.evaluation.metrics.base import BaseMetric

class BrierScoreMetric(BaseMetric):
    """Computes Brier score for probabilistic predictions.
    
    The Brier score measures the accuracy of probabilistic predictions.
    It is the mean squared difference between the predicted probability
    and the actual outcome.
    
    Args:
        device (str): Device to perform computations on ('cpu' or 'cuda').
    """
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.reset()
        
    def reset(self) -> None:
        """Reset Brier score computation."""
        self.total_score = 0.0
        self.num_samples = 0  # Changed from num_batches to track total samples
        
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update Brier score with batch predictions.
        
        Args:
            logits (torch.Tensor): Raw output logits from the model 
                                  [batch_size, seq_length, vocab_size]
            targets (torch.Tensor): Target token indices 
                                   [batch_size, seq_length]
        """
        # Process logits and targets
        shifted_logits = logits[:, :-1].contiguous()
        shifted_targets = targets[:, 1:].contiguous()
        
        # Flatten and filter
        flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
        flat_targets = shifted_targets.view(-1)
        
        # Filter out padding tokens (usually marked as -100)
        valid_mask = flat_targets != -100
        
        # Skip update if no valid tokens in batch
        if not valid_mask.any():
            return
            
        valid_logits = flat_logits[valid_mask]
        valid_targets = flat_targets[valid_mask]
        
        # Verify valid targets are within the vocabulary size
        if torch.max(valid_targets) >= valid_logits.size(-1):
            raise ValueError(
                f"Target contains indices {torch.max(valid_targets).item()} that are "
                f"out of bounds for the vocabulary size {valid_logits.size(-1)}"
            )
        
        # Compute probabilities and one-hot targets
        probs = F.softmax(valid_logits, dim=-1)
        target_onehot = F.one_hot(valid_targets, num_classes=probs.size(-1)).float()
        
        # Calculate squared error for each prediction
        squared_error = (probs - target_onehot) ** 2
        
        # Sum the squared error across all predictions
        batch_score = torch.sum(squared_error).item()
        
        # Count valid predictions
        num_valid = valid_mask.sum().item()
        
        # Update running totals
        self.total_score += batch_score
        self.num_samples += num_valid
        
    def compute(self) -> float:
        """Compute final Brier score.
        
        Returns:
            float: Average Brier score across all samples.
        """
        if self.num_samples == 0:
            return 0.0
        return self.total_score / self.num_samples
