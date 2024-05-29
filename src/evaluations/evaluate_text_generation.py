import logging
import torch
import torchmetrics

pruna_logger = logging.getLogger("pruna_logger")


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device="cuda"):
    if isinstance(model, torch.nn.Module):
        model.eval()

    model.to(device)

    metric = torchmetrics.text.Perplexity(ignore_index=-100).to(device)  # -100 is the padding token.

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x).logits

        # Metric on current batch
        perplexity = metric(logits.float(), y)

    # Metric on all batches using custom accumulation
    perplexity = metric.compute()

    torch.cuda.empty_cache()
    return perplexity.item()
