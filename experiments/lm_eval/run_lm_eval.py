import os
from dotenv import load_dotenv

load_dotenv(override=True)
if os.environ.get("HF_HOME"):
    hf_home = os.environ["HF_HOME"]
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_home
    os.environ["HUGGINGFACE_ASSETS_CACHE"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import hydra
from omegaconf import DictConfig
import torch
import logging
from datetime import datetime
from pathlib import Path

from huggingface_hub import login
import lm_eval
from lm_eval.utils import setup_logging as lm_eval_setup_logging

from src.loggers.setup_logging import setup_logging
from src.model_loading.registry.enhanced_registry import EnhancedModelRegistry
from src.model_loading.registry.registry import ModelRegistry
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.manager import ModelManager
from src.model_loading.loaders.hqq import HQQModelLoader
from src.model_loading.loaders.quanto import QuantoModelLoader
from src.model_loading.loaders.gptq import GPTQModelLoader
from src.model_loading.loaders.awq import AWQModelLoader
from src.model_loading.loaders.bitsandbytes import BitsAndBytesModelLoader
from src.model_loading.loaders.standard import StandardModelLoader

from lm_eval_wrapper import CustomModelWrapper

logger = setup_logging()
lm_eval_setup_logging("WARNING")

# Maps our dataset type names to lm-eval task strings
TASK_MAP = {
    "triviaqa":       "triviaqa",
    "coqa":           "coqa",
    "commonsenseqa":  "commonsense_qa",
    "arc_easy":       "arc_easy",
    "mmlu":           "mmlu",
    "hellaswag":      "hellaswag",
    "piqa":           "piqa",
    "race":           "race",
    "ceval":          "ceval-valid",
    "gsm8k":          "gsm8k",
    "lambada":        "lambada",
}


def authenticate_huggingface() -> None:
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token is None:
        raise ValueError("HUGGINGFACE_TOKEN not set. Copy .env.example to .env and fill in your token.")
    os.environ["HF_TOKEN"] = token
    login(token=token, add_to_git_credential=False)
    logger.info("HuggingFace authentication successful.")


def load_model(model_name: str, cfg: DictConfig) -> CustomModelWrapper:
    base_registry = ModelRegistry()
    model_registry = EnhancedModelRegistry(base_registry)

    identifier = model_registry.get_model_by_string(model_name)
    if identifier is None:
        raise ValueError(f"Model not found in registry: {model_name}")

    paths = base_registry.get_model_paths(identifier)
    config = ModelConfig(
        identifier=identifier,
        device=cfg.hardware.device if cfg.hardware.device != "auto" else "cuda",
        trust_remote_code=True,
        apply_compile=False,
        cache_dir=os.environ.get("HF_HOME", "~/.cache/huggingface"),
    )
    config.model_path = paths.model_path
    config.tokenizer_path = paths.tokenizer_path

    if cfg.hardware.device == "auto":
        config.device_map = "auto"
        max_memory = cfg.hardware.get("max_memory")
        if max_memory:
            config.max_memory = {int(k): v for k, v in max_memory.items()}

    name_lower = model_name.lower()
    if "gptq" in name_lower:
        loader = GPTQModelLoader()
        loader_type = "gptq"
    elif "awq" in name_lower:
        loader = AWQModelLoader()
        loader_type = "awq"
    elif "hqq" in name_lower:
        loader = HQQModelLoader()
        loader_type = "hqq"
    elif "quanto" in name_lower:
        loader = QuantoModelLoader()
        loader_type = "quanto"
    elif "bnb" in name_lower:
        loader = BitsAndBytesModelLoader()
        loader_type = "bnb"
    else:
        loader = StandardModelLoader()
        loader_type = "hf"

    ModelManager()._validate_model_cache(loader, config)
    model, tokenizer = loader.load_model(config)

    if loader_type == "gptq":
        _float_dtypes = (torch.float16, torch.float32, torch.bfloat16)
        for p in model.parameters():
            if p.dtype in _float_dtypes:
                p.data = p.data.to(config.device)
        for b in model.buffers():
            if b.dtype in _float_dtypes:
                b.data = b.data.to(config.device)

    is_multimodal = any(k in name_lower for k in ["qwen3vl", "qwen25vl"])
    processor = None
    actual_tokenizer = tokenizer
    if is_multimodal and hasattr(tokenizer, "tokenizer"):
        processor = tokenizer
        actual_tokenizer = tokenizer.tokenizer

    return CustomModelWrapper(
        model=model,
        tokenizer=actual_tokenizer,
        model_name=model_name,
        batch_size=1,
        is_multimodal=is_multimodal,
        processor=processor,
    )


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    authenticate_huggingface()

    task_key = cfg.task.name
    lm_eval_task = TASK_MAP.get(task_key, task_key)

    if "ceval" in lm_eval_task or "bigbench" in lm_eval_task:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_dir = Path(cfg.output_dir) / cfg.exp_id / lm_eval_task / cfg.model_name.replace("_", "-")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {cfg.model_name}  Task: {lm_eval_task}  Limit: {cfg.num_entries}")

    wrapped_model = load_model(cfg.model_name, cfg)

    task_manager = lm_eval.tasks.TaskManager()
    evaluation_tracker = lm_eval.loggers.evaluation_tracker.EvaluationTracker(
        output_path=str(output_dir)
    )

    results = lm_eval.simple_evaluate(
        model=wrapped_model,
        tasks=lm_eval_task.split(","),
        num_fewshot=cfg.num_fewshot,
        task_manager=task_manager,
        evaluation_tracker=evaluation_tracker,
        limit=cfg.num_entries if isinstance(cfg.num_entries, int) else None,
        batch_size=1,
        write_out=True,
        log_samples=True,
        confirm_run_unsafe_code=True,
    )

    samples = results.pop("samples")

    for t_name, t_samples in samples.items():
        for sample in t_samples:
            key = (t_name, sample.get("doc_id"))
            val = wrapped_model._generation_logprobs.get(key)
            if val is not None:
                sample["generation_mean_logprob"], sample["generation_mean_entropy"] = val

    evaluation_tracker.date_id = datetime.now().isoformat().replace(":", "-")
    evaluation_tracker.save_results_aggregated(results=results, samples=samples)
    for t_name, t_samples in samples.items():
        evaluation_tracker.save_results_samples(task_name=t_name, samples=t_samples)

    logger.info(f"Results saved to {output_dir}")
    for task, metrics in results.get("results", {}).items():
        logger.info(f"  {task}: {metrics}")

    base_registry = ModelRegistry()
    model_registry = EnhancedModelRegistry(base_registry)
    try:
        identifier = model_registry.get_model_by_string(cfg.model_name)
        paths = base_registry.get_model_paths(identifier)
        cleanup_cfg = ModelConfig(identifier=identifier, device="cuda", apply_compile=False)
        cleanup_cfg.model_path = paths.model_path
        cleanup_cfg.tokenizer_path = paths.tokenizer_path
        ModelManager().cleanup_model_cache(cleanup_cfg)
    except Exception as e:
        logger.warning(f"Cache cleanup failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
