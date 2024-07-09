import logging
import torch
import torchmetrics
import tqdm

logger = logging.getLogger("quant_logger")

with torch.no_grad():
    torch.cuda.empty_cache()

def evaluate_perplexity(model, dataloader, factor=1, device="cuda", to_device=False):
    if to_device:
        model.to(device)
    if isinstance(model, torch.nn.Module):
        model.eval()
        print(f"Model in evaluation mode. Device: {device}")
    metric = torchmetrics.text.Perplexity(ignore_index=-100).to(device)  # -100 is the padding token.

    for i, (x, y) in enumerate(dataloader):
        if i >= len(dataloader) / factor:
            break
        print(f"Processing batch {i}")
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            outputs = model(x)
            logits = outputs.logits
            
            # Metric on current batch
            perplexity = metric(logits.float(), y)

    # Metric on all batches using custom accumulation
    perplexity = metric.compute()
    print(f"\nFinal Perplexity (PPL): {perplexity:.3f}")
    return perplexity.item()
