import logging
import torch
import torchmetrics

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from netcal.presentation import ReliabilityDiagram
import numpy as np
from netcal.metrics import ECE

import numpy as np
from sklearn import metrics as skm

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
    aurc = area_under_risk_coverage_score(y_confs, y_true)
    result_matrics['aurc'] = aurc
    print("AURC score:", aurc)

    # ECE
    n_bins = 10
    # diagram = ReliabilityDiagram(n_bins)
    ece = ECE(n_bins)
    ece_score = ece.measure(np.array(y_confs), np.array(y_true))
    print("ECE:", ece_score)
    result_matrics['ece'] = ece_score

    return result_matrics
