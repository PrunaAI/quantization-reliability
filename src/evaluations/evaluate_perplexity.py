import logging
import torch
import torchmetrics

logger = logging.getLogger("quant_logger")

def evaluate_perplexity(model, dataloader, device="cuda", to_device=False, verbose=False):
    if to_device:
        model.to(device)
    if isinstance(model, torch.nn.Module):
        model.eval()
        logger.info(f"Model in evaluation mode. Device: {device}")
        
    metric = torchmetrics.text.Perplexity(ignore_index=-100).to(device)  # -100 is the padding token.
    for i, (x, y) in enumerate(dataloader):
        if verbose:
            logger.info(f"Processing batch {i + 1}/{len(dataloader)}")
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            outputs = model(x)
            logits = outputs.logits
            
            # Metric on current batch
            perplexity = metric(logits.float(), y)

    # Metric on all batches using custom accumulation
    perplexity = metric.compute()
    logger.info(f"Final Perplexity: {perplexity:.3f}")
    return perplexity.item()
