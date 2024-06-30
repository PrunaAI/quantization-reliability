import torch
import tqdm
import torch.nn.functional as F

def evaluate_brier_score(model, dataloader, device="cuda"):
    if isinstance(model, torch.nn.Module):
        model.eval()

    model.to(device)

    metric = torchmetrics.text.Perplexity(ignore_index=-100).to(device)  # -100 is the padding token.

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x).logits

        # Metric on current batch
        brier_score = (y-logits) ** 2 #TODO

    # Metric on all batches using custom accumulation
    perplexity = metric.compute()

    torch.cuda.empty_cache()
    return perplexity.item()

def evaluate_brier_score(model, tokenizer, dataloader, max_length=None, stride=512, factor=1, to_device=False, device="cuda"):
    if max_length is None:
        max_length = tokenizer.model_max_length
    if to_device:
        model.to(device)
        
    encodings = tokenizer("\n\n".join(dataloader.dataset.dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    brier_scores = []
    prev_end_loc = 0
    for begin_loc in tqdm.tqdm(range(0, seq_len//factor, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone().to(device)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            logits = outputs.logits
            
            # Shift logits and target_ids to the left by 1 for calculating the Brier score
            shifted_logits = logits[:, :-1].contiguous()
            shifted_target_ids = target_ids[:, 1:].contiguous()

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

            # Calculate the Brier score
            brier_score = torch.mean((probs - targets) ** 2)
            brier_scores.append(brier_score)

    avg_brier_score = torch.stack(brier_scores).mean()
    print(f"Brier Score of model {model.NAME}: {avg_brier_score:.4f}")
    
    return avg_brier_score