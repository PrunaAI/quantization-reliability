import logging
import torch
import torchmetrics

pruna_logger = logging.getLogger("quant_logger")

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device="cuda", send_to_device=False, logger_name="quant_logger"):
    # Configure logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # Adjust logging level as needed
    
    try:
        if isinstance(model, torch.nn.Module):
            model.eval()

        if send_to_device:
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
        logger.info(f"Successfully computed perplexity for model: {model.name_or_path}")
    
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        raise e  # Re-raise the exception
    
    return perplexity.item()
