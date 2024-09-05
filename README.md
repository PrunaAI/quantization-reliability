# Reliability of Extreme Quantization Methods for Large Language Models

## Introduction

This project aims to evaluate the reliability of large language models (LLMs) when subjected to various quantization methods. Our primary focus is on assessing the reliability of the Llama-3-8B model using a range of quantization techniques. The ultimate goal is to identify potential scaling laws for the reliability of quantized models, similar to those established for sparse foundation models.

## Project Goals

1. **Evaluate the Reliability of Quantized LLMs**: Assess the reliability of the Llama-3-8B model when quantized using various methods.
2. **Zero-Shot Task Performance**: Measure the model's confidence and performance on zero-shot tasks.
3. **Scaling Laws**: Investigate potential scaling laws for the reliability of quantized LLMs.
4. **Calibration and Evaluation**: Use metrics like perplexity, Brier score, and others to evaluate model performance on both in-distribution (ID) and out-of-distribution (OOD) datasets.
5. **Input Perturbation Analysis**: Evaluate the robustness of neural language models to various input perturbations.

## Current Focus

- Implementing and evaluating prompting strategies for reliability assessment.
- Analyzing the impact of different `max_new_tokens` values on model performance.
- Evaluating model performance under various taxonomies and typo-based perturbations.
- Assessing the reliability of quantized models using perplexity and Brier score metrics.

## Methods Used

### Quantization Methods

- BnB (Bitsandbytes)
- AWQ (Activation-aware Weight Quantization)
- HQQ (Hierarchical Quantization)
- HQQ+
- Quanto
- Quanto+
- QuaRoT (Quantized Approximation of Rotational Transformations)
- QoQ (Quattuor-Octo-Quattuor)
- SpQR (Sparse-Quantized Representation)
- AQLM (Additive Quantization of Language Models)
- GPTQ (Generalized Quantization)

### PEFT Methods

- LoftQ (LoRA-Fine-Tuning-aware Quantization)
- QLoRA (Quantization-aware LoRA)
- LoRA (Low-Rank Adaptation)

### Evaluation Metrics

- Perplexity
- Brier Score
- AUCPR (Area Under the Precision-Recall Curve)
- Accuracy

### Datasets

- WikiText
- C4
- PTB (Penn Treebank)
- OpenAssistant
- FKTC (Factual Knowledge and Taxonomic Classification)

## Repository Structure

- `notebooks/`: Contains Jupyter notebooks with examples and experiments.
- `scripts/`: Contains the main scripts for running experiments.
- `src/`: Contains the core source code of the project.

## Developer Pipeline

### Initial Setup

1. Create a new conda environment with Python 3.10:
   ```
   conda create -n quant-rel python=3.10
   ```

2. Install CUDA (if not already installed):
   ```
   conda install nvidia/label/cuda-11.8.0::cuda
   ```

3. Install the project in editable mode:
   ```
   pip install -e .
   ```

### Code Formatting and Checking

- Use pre-commit with black and flake8 for code formatting and linting.
- Install pre-commit: `pip install pre-commit`
- Set up the `.pre-commit-config.yaml`, `pyproject.toml`, and `.flake8` files as described in the original README.

### Experiment Tracking and Scheduling

- Use Weight & Biases for experiment tracking.
- Use SEML for scheduling experiments on the server.

## Future Work

- Complete the evaluation pipeline for all quantization methods.
- Implement and evaluate PEFT methods in combination with quantization.
- Conduct a comprehensive analysis of model robustness to input perturbations.
- Investigate scaling laws for quantized models across different model sizes and quantization methods.

## Acknowledgements

This project is being conducted at the Technical University of Munich in collaboration with Pruna AI, under the supervision of Bertrand Charpentier.
