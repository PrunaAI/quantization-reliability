import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("quant_logger")

class BrierScore:
    def __init__(self, device="cpu"):
        self.device = device
        self.reset()

    def reset(self):
        self.total_brier_score = 0.0
        self.num_batches = 0

    def update(self, probs, targets):
        brier_score = torch.mean((probs - targets) ** 2)
        self.total_brier_score += brier_score.item()
        self.num_batches += 1

    def compute(self):
        if self.num_batches == 0:
            return 0.0
        return self.total_brier_score / self.num_batches

def evaluate_brier_score(model, dataloader, n_samples=None, device="cuda", to_device=False, verbose=False):
    if n_samples is None:
        n_samples = len(dataloader)
    if to_device:
        model.to(device)
    if isinstance(model, torch.nn.Module):
        model.eval()
        logger.info(f"Model in evaluation mode. Device: {device}")
    
    # Initialize BrierScore metric
    metric = BrierScore(device=device)
    for i, (x, y) in enumerate(dataloader):
        if i >= n_samples:
            break
        if verbose:
            logger.info(f"Processing batch {i + 1}/{len(dataloader)}")
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            outputs = model(x)
            logits = outputs.logits

            # Shift logits and target_ids to the left by 1 for calculating the Brier score
            shifted_logits = logits[:, :-1].contiguous()
            shifted_target_ids = x[:, 1:].contiguous()

            # Flatten the logits and target_ids for calculation
            shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))
            shifted_target_ids = shifted_target_ids.view(-1)

            # Filter out the -100 targets
            valid_indices = shifted_target_ids != -100
            valid_logits = shifted_logits[valid_indices]
            valid_target_ids = shifted_target_ids[valid_indices]

            # Get the probabilities
            probs = F.softmax(valid_logits, dim=-1)

            # Create one-hot target vectors
            targets = F.one_hot(valid_target_ids, num_classes=probs.size(-1)).float()

            # Update the metric with the current batch's results
            metric.update(probs, targets)

    # Compute the final Brier score across all batches
    avg_brier_score = metric.compute()
    logger.info(f"Final Brier Score: {avg_brier_score:.6f}")
    return avg_brier_score
