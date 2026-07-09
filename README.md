# Reliability Evaluation of Extreme Quantization for Large Language Models

[[Paper]](https://openreview.net/forum?id=jehhhDHum2&nesting=2&sort=date-desc)

Code for the paper *"Reliability Evaluation of Extreme Quantization for Large Language Models"*, published in Transactions on Machine Learning Research (TMLR), 2025.

We evaluate how extreme weight quantization (1–8 bit) affects not just accuracy but calibration and uncertainty estimation across multiple model families, quantization methods, and benchmarks.

---


---

## Experiments

The paper uses two evaluation pipelines:

| Pipeline | Datasets | Metrics |
|----------|----------|---------|
| **Reliability eval** (custom) | TriviaQA, CoQA, CommonsenseQA | AUCPR, AUCROC, Brier, NLL, entropy |
| **lm-eval harness** | C-Eval, RACE, PIQA, ARC-Easy, Hellaswag, MMLU | accuracy |

### Reliability evaluation

Evaluates calibration, NLL, entropy, and AUCPR/AUCROC across quantized models. Supports text perturbations (15 char- and word-level types at 3 intensity levels).

```bash
cd experiments/

# single run
python run.py hardware=single_gpu dataset=triviaqa model_name=llama3_8b exp_id=test

# predefined sweep (full model family × dataset × perturbations)
bash sweeps/llama32_1b_triviaqa.sh
```

### KL divergence evaluation

Measures output distribution shift between full-precision and quantized models.

```bash
cd experiments/kld/

# single run
python run_kld.py hardware=single_gpu dataset=c4 model_name=llama3_8b exp_id=test

# predefined sweep
bash sweeps/llama_kld.sh
```

### Accuracy evaluation (lm-eval harness)

Standard benchmark accuracy is measured via [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness). See `experiments/lm_eval/` for the task configs and run instructions.

### Configuration (Hydra)

Reliability and KLD experiments use [Hydra](https://hydra.cc/) for config composition. Configs are composed from three groups:

| Group | Location | Options |
|-------|----------|---------|
| `hardware` | `configs/hardware/` | `single_gpu`, `multi_gpu_2`, `multi_gpu_4`, `multi_gpu_5`, `multi_gpu_7`, `cpu` |
| `dataset` | `configs/dataset/` | `triviaqa`, `triviaqa_perturbed`, `coqa`, `commonsenseqa` |
| shared params | `configs/config.yaml` | `model_name`, `temperature`, `num_entries`, ... |

Any parameter can be overridden inline:

```bash
python run.py hardware=single_gpu dataset=triviaqa model_name=llama32_1b \
  exp_id=my-run temperature=0.5 use_wandb=false
```

Multirun sweeps use Hydra's `--multirun` flag:

```bash
python run.py --multirun hardware=single_gpu dataset=triviaqa \
  exp_id=sweep model_name=llama32_1b,llama32_1b_hqq_4bit,llama32_1b_gptq_4bit
```

---

## Project structure

```
experiments/
    run.py                      # reliability eval entry point
    configs/                    # Hydra configs for reliability eval
    sweeps/                     # sweep scripts per model family
    kld/                        # KLD eval (own entry point, configs, sweeps)
    lm_eval/                    # lm-evaluation-harness task configs + instructions
src/
    dataset_processing/         # dataset loading, perturbations
    model_loading/              # model registry, loaders (HQQ, GPTQ, AWQ, ...)
    reliability_eval/           # pipelines: NLL, confidence, entropy, top-k
    loggers/
notebooks/
    plots/paper/                # scripts to reproduce all paper figures
data/
    README.md                   # data format and download instructions
```

---

## Supported models and methods

**Model families:** LLaMA 3 / 3.1 / 3.2 (1B–70B), OPT (125M–13B), Qwen3 (4B–32B)

**Quantization methods:** BitsAndBytes, GPTQ, HQQ, AWQ, Quanto, AQLM, QuaRoT, QoQ

**Reliability eval datasets:** TriviaQA, CoQA, CommonsenseQA

**Accuracy eval datasets:** C-Eval, RACE, PIQA, ARC-Easy, Hellaswag, MMLU

**Perturbations:** 15 character- and word-level text perturbations at 3 intensity levels

---

## Citation

```bibtex
@article{daroczi2025reliability,
  title     = {Reliability Evaluation of Extreme Quantization for Large Language Models},
  author    = {D\'aroczi, S\'andor and Charpentier, Bertrand and Ayadi, Sirine},
  journal   = {Transactions on Machine Learning Research},
  year      = {2025},
  url       = {TODO-openreview-link}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
