# Project

## Introduction

This project aims to evaluate the reliability of large language models (LLMs) when subjected to various quantization methods. Our first focus is on assessing the zero-shot reliability of the Llama-3-8B model using methods like BnB, AWQ, HQQ, GPTQ, QuARoT, and many more. The ultimate goal is to identify scaling laws for the reliability of quantized models, similar to those established for sparse foundation models. 

## Project Goals

1. **Evaluate the Reliability of Quantized LLMs**: Assess the reliability of the Llama-3-8B model when quantized using various methods.
2. **Zero-Shot Task Performance**: Measure the model's confidence and performance on zero-shot tasks.
3. **Scaling Laws**: Investigate potential scaling laws for the reliability of quantized LLMs.
4. **Calibration and Evaluation**: Use metrics like perplexity, Brier score, and others to evaluate model performance on both in-distribution (ID) and out-of-distribution (OOD) datasets.


## Repository structure

- `notebooks` contains the notebooks including examples.
- `scripts` contains the main scripts.
- `src` contains the core source code .

# Developper pipeline

- Setup "Implementation pipeline" (incl. "Initial Setup" and "Code formatting and checking")
- Comment your code using Numpy/Scipy Docstrings styles (see [here](https://numpydoc.readthedocs.io/en/latest/format.html))

## Initial setup

### Creating a New Conda Environment
Create a new conda env with one python version in 3.8, 3.9, 3.10 by running `conda create -n myenv python=3.X`.

### Installing CUDA & packaging
Make sure that you have cuda installed with `nvcc --version` or do `conda install nvidia/label/cuda-12.1.0::cuda` for CUDA 12 or `conda install nvidia/label/cuda-11.8.0::cuda` for CUDA 11.

### Install for dev
Navigate to the src (where the `setup.py` is located) folder and then install using:

```shell
pip install -e .
```

## Format and check code with pre-commit running black and flake8

- Install `pip install pre-commit`
- Define `.pre-commit-config.yaml` with the hooks you want to include. See first example [here](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
- Define `pyproject.toml` at root with `line-length=121`. See example [here](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
- Define `.flake8` at root with `max-line-length=121`. See example [here](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
- Run `pre-commit install`
- You can try it with `pre-commit run --all-files`.

## Track experiments with Weight&Biases

- Install Weight&Biases.
- Authorize your API key using `wandb login`.
- Create and use the porject id in Weight&Biases.

For more detail see [official instructions](https://wandb.ai/quickstart/pytorch).

## Schedule experiments with seml or ray-tune

For more detail on seml see [official instructions](https://github.com/TUM-DAML/seml).

For more detail on ray-tune see [official instructions](https://docs.ray.io/en/latest/tune/index.html).

## Methods Used

### Quantization Methods

BnB: Bitsandbytes quantization.
AWQ: Adaptive Weight Quantization.
HQQ: Hierarchical Quantization.
GPTQ: Generalized Quantization.
QuARoT: Quantized Approximation of Rotational Transformations.

### Evaluation Metrics
Perplexity: Evaluated using datasets like WikiText.
Brier Score: Evaluated using torch.metrics or HuggingFace.

### Datasets
Calibration Dataset: WikiText, OpenAssistantDataModule.
Evaluation Dataset: WikiText, OpenAssistantDataModule.
