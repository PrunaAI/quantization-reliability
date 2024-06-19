import logging
import torch
from torchmetrics import BrierScore
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from netcal.metrics import ECE

import numpy as np

pruna_logger = logging.getLogger("quant_logger")

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, data_module, device="cuda"):
    encodings = tokenizer("\n\n".join(data_module.train_dataset["text"]), return_tensors="pt")
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
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
    print(f"Perplexity of model {model.name_or_path}: {ppl:.2f}")
    
    return ppl

@torch.no_grad()
def evaluate_brier_score(model, tokenizer, data_module, device="cuda"):
    encodings = tokenizer("\n\n".join(data_module.train_dataset["text"]), return_tensors="pt")
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    brier_score_metric = BrierScore(num_classes=tokenizer.vocab_size).to(device)
    
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

            for i in range(input_ids.size(1)):
                actual = target_ids[:, i]
                predicted = probabilities[:, i, :]
                brier_score_metric.update(predicted, actual)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_brier_score = brier_score_metric.compute().item()
    print(f"Brier Score of model {model.name_or_path}: {avg_brier_score:.4f}")
    
    return avg_brier_score

def compute_conf_metrics(y_true, y_confs):
    result_matrics = {}
    # ACC
    accuracy = sum(y_true) / len(y_true)
    print("accuracy: ", accuracy)
    result_matrics['acc'] = accuracy

    # use np to test if y_confs are all in [0, 1]
    assert all([x >= 0 and x <= 1 for x in y_confs]), y_confs
    y_confs, y_true = np.array(y_confs), np.array(y_true)
    
    # AUCROC
    roc_auc = roc_auc_score(y_true, y_confs)
    print("ROC AUC score:", roc_auc)
    result_matrics['auroc'] = roc_auc

    # AUPRC-Positive
    auprc = average_precision_score(y_true, y_confs)
    print("AUC PRC Positive score:", auprc)
    result_matrics['auprc_p'] = auprc

    # AUPRC-Negative
    auprc = average_precision_score(1- y_true, 1 - y_confs)
    print("AUC PRC Negative score:", auprc)
    result_matrics['auprc_n'] = auprc
    
    # AURC from https://github.com/IML-DKFZ/fd-shifts/tree/main
    # aurc = area_under_risk_coverage_score(y_confs, y_true)
    # result_matrics['aurc'] = aurc
    # print("AURC score:", aurc)

    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs), np.array(y_true))
    print("ECE:", ece_score)
    result_matrics['ece'] = ece_score

    return result_matrics
