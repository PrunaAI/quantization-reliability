from src.evaluations.evaluate_brier_score import evaluate_brier_score
from src.evaluations.evaluate_perplexity import (
    evaluate_perplexity
)

# Evaluating reliability
evaluate_metrics = {
    "perplexity": evaluate_perplexity,
    "brier_score": evaluate_brier_score,
    # TODO: new metrics for reliability
}

name_metrics = {
    "accuracy": "Accuracy (%)",
    "perplexity": "Perplexity",
    "memory_disk_first": "Disk memory first (MB)",
    "memory_disk": "Disk memory (MB)",
    "memory_inference_first": "Inference memory first (MB)",
    "memory_inference": "Inference memory (MB)",
    "memory_training": "Training memory (MB)",
    "token_generation_latency_sync": "Sync. token gen. latency (ms/token)",
    "token_generation_latency_async": "Async. Token gen. latency (ms/token)",
    "token_generation_throughput_sync": "Sync. Token gen. throughput (token/ms)",
    "token_generation_throughput_async": "Async. Token gen. throughput (token/ms)",
    "token_generation_CO2_emissions": "Token gen. CO2 emissions (KgCO2e/token)",
    "token_generation_energy_consumption": "Token gen. energy consumption (KWh/token)",
    "inference_latency_sync": "Sync. inference latency (ms/pass)",
    "inference_latency_async": "Async. inference latency (ms/pass)",
    "inference_throughput_sync": "Sync. inference throughput (pass/ms)",
    "inference_throughput_async": "Async. inference throughput (pass/ms)",
    "inference_money_cost": "Inference money cost ($/pass)",
    "inference_CO2_emissions": "Inference CO2 emissions (KgCO2e)",
    "inference_energy_consumption": "Inference energy consumption (KWh)",
    "inference_macs": "Inference MACs (G)",
    "inference_deviation_bleu": "Inference deviation Bleu (%)",
    "inference_deviation_norm": "Inference deviation norm (%)",
}
