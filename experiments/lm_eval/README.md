# Accuracy Evaluation (lm-evaluation-harness)

This directory contains the **accuracy** evaluation pipeline, which measures
standard benchmark accuracy (not calibration) using a custom wrapper around
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).

---

## How this differs from `experiments/run.py`

| | This pipeline (`lm_eval/`) | Reliability pipeline (`experiments/run.py`) |
|---|---|---|
| **What it measures** | Accuracy on standard benchmarks | Calibration: AUCPR, AUCROC, Brier, NLL, entropy |
| **Datasets** | C-Eval, ARC-Easy, Hellaswag, PIQA, RACE, TriviaQA, CoQA, CSQA | TriviaQA, CoQA, CommonsenseQA |
| **Perturbations** | No — lm-eval manages dataset internals | Yes — 15 char/word-level types at 3 intensities |
| **Model loading** | Same custom loaders via `CustomModelWrapper` | Same custom loaders via `src/model_loading/` |
| **Output** | JSON accuracy results per task | Excel/CSV with per-sample reliability metrics |

TriviaQA, CoQA, and CommonsenseQA appear in **both** pipelines because the paper
reports both accuracy (here) and calibration metrics (in `experiments/`).
C-Eval, ARC-Easy, Hellaswag, PIQA, and RACE are **only** evaluated for accuracy
and do not have a reliability eval counterpart — they are classification/multiple-choice
tasks where the custom calibration pipeline does not apply.

---

## Usage

```bash
cd experiments/lm_eval/

# single run
python run_lm_eval.py hardware=single_gpu task=ceval model_name=llama3_8b exp_id=test

# sweep
bash sweeps/llama_accuracy.sh
```

Available tasks: `triviaqa`, `coqa`, `commonsenseqa`, `ceval`, `arc_easy`, `hellaswag`, `piqa`, `race`

Any parameter can be overridden inline:

```bash
python run_lm_eval.py hardware=single_gpu task=arc_easy model_name=llama32_1b \
  exp_id=test num_entries=100
```

---

## Output

Results are saved to `../../lm_eval_harness_results/<exp_id>/<task>/<model>/`.
Each run produces aggregated JSON (accuracy metrics) and per-sample JSON
(including generation log-probs and entropy captured as a side channel).
