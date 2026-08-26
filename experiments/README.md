# Experiments

This directory contains three evaluation pipelines. They measure different things
and are run independently.

| Directory | Entry point | What it measures | Datasets |
|-----------|-------------|-----------------|--------------|
| `./` (root) | `run.py` | **Calibration / reliability**: AUCPR, AUCROC, Brier, NLL, entropy | TriviaQA, CoQA, CommonsenseQA |
| `lm_eval/` | `lm_eval/run_lm_eval.py` | **Accuracy** on standard benchmarks | C-Eval, ARC-Easy, Hellaswag, PIQA, RACE, TriviaQA, CoQA, CSQA |
| `kld/` | `kld/run_kld.py` | **KL divergence** between fp and quantized model outputs | C4, Wikitext |

All three pipelines use the same model registry and loaders from `src/model_loading/`.
All use Hydra for config composition. The evaluated configurations are under `configs/`.
