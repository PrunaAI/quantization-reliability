import torch
import tqdm

def evaluate_perplexity(model, tokenizer, dataloader, max_length=None, stride=512, factor=1, to_device=False, device="cuda"):
    if max_length is None:
        max_length = tokenizer.model_max_length
    if to_device:
        model.to(device)
        
    encodings = tokenizer("\n\n".join(dataloader.dataset.dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm.tqdm(range(0, seq_len//factor, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone().to(device)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity of model {model.NAME}: {ppl:.2f}")
    
    return ppl