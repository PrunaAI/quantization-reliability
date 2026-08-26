import os
import sys
from dotenv import load_dotenv

# Cache paths must be set before any HuggingFace import
load_dotenv(override=True)
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME)
os.environ.setdefault("HUGGINGFACE_ASSETS_CACHE", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import random
import time
from datetime import datetime

import numpy as np
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import login

torch.hub.set_dir(HF_HOME)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_processing.common.dataset_types import DatasetType
from src.dataset_processing.common.source_types import DatasetSourceType
from src.model_loading.registry.enhanced_registry import EnhancedModelRegistry
from src.model_loading.registry.registry import ModelRegistry
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.manager import ModelManager
from src.reliability_eval.common.mappings import (
    DATASET_TYPE_MAP, GENERATION_STRATEGY_MAP, PERTURBATION_TYPE_MAP,
    PIPELINE_TYPE_MAP, PROMPT_STRATEGY_MAP, SOURCE_TYPE_MAP,
)
from src.reliability_eval.pipeline.evaluation_pipelines.registry import EVALUATION_PIPELINES_DICT
from src.reliability_eval.pipeline.processor.results_processor import post_process_results
from src.reliability_eval.pipeline.evaluation_pipelines.types import PipelineType
from src.reliability_eval.common.experiment import GenerationExperimentConfig
from src.reliability_eval.pipeline.config import BatchConfig, IntegratedPipelineConfig
from src.reliability_eval.pipeline.core import IntegratedEvaluationPipeline
from src.dataset_processing.dataset_factory import DatasetFactory
from src.loggers.setup_logging import setup_logging

logger = setup_logging()


def authenticate_huggingface() -> None:
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not set in .env or environment")
    login(token=token, add_to_git_credential=True)
    logger.info("Authenticated with HuggingFace")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_batch_size(model_name: str, dataset_type: str, num_repeats: int, vram_gb: int) -> int:
    batch_size = 32
    is_small_gpu = vram_gb < 20
    if is_small_gpu and dataset_type == "COQA":
        batch_size = 8 if any(x in model_name for x in ("llama32_1b", "llama32_3b")) else 4
    if not is_small_gpu and dataset_type == "COQA":
        batch_size //= 2
    if not is_small_gpu and "opt_" in model_name:
        batch_size //= 2
    if "_13b" in model_name:
        batch_size //= 4
    if num_repeats > 1:
        batch_size //= num_repeats
    return batch_size


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict:
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    authenticate_huggingface()
    set_seeds(42)

    # AQLM only supports greedy search
    generation_strategy = "greedy_search" if "aqlm" in cfg.model_name.lower() else "multinomial_sampling"

    batch_size = cfg.batch_size
    if batch_size == "auto":
        batch_size = get_batch_size(cfg.model_name, cfg.dataset.type, 1, cfg.hardware.vram_gb)

    # Convert max_memory keys from str to int (YAML forces string keys)
    max_memory = OmegaConf.to_container(cfg.hardware.max_memory) if cfg.hardware.max_memory else None
    if max_memory:
        max_memory = {int(k): v for k, v in max_memory.items()}

    # Convert strings to enums
    dataset_type_enum = DATASET_TYPE_MAP[cfg.dataset.type]
    source_type_str = "raw" if cfg.dataset.perturbation_type == "none" else "processed"
    source_type_enum = SOURCE_TYPE_MAP[source_type_str]
    perturbation_type_enum = PERTURBATION_TYPE_MAP[cfg.dataset.perturbation_type]
    prompt_strategy_enum = PROMPT_STRATEGY_MAP["Original"]
    generation_strategy_enum = GENERATION_STRATEGY_MAP[generation_strategy]
    pipeline_types_enum = [
        PIPELINE_TYPE_MAP["nll_pipeline"],
        PIPELINE_TYPE_MAP["confidence_pipeline"],
        PIPELINE_TYPE_MAP["entropy_pipeline"],
        PIPELINE_TYPE_MAP["topk_pipeline"],
    ]

    # Dataset config
    config_class = DatasetFactory.create_config(dataset_type_enum)
    dataset_config_dict = {
        "dataset_type": dataset_type_enum,
        "dataset_name": cfg.dataset.type.lower(),
        "source_type": source_type_enum,
        "num_entries": cfg.num_entries,
        "num_shots": 0,
        "force_reprocess": False,
        "merge_config": None,
        "random_seed": 42,
    }
    if source_type_enum == DatasetSourceType.PROCESSED:
        dataset_config_dict["perturbation_type"] = perturbation_type_enum
        dataset_config_dict["perturbation_intensity"] = cfg.dataset.perturbation_intensity
    dataset_config = config_class(**dataset_config_dict)

    # Model registry
    base_registry = ModelRegistry()
    model_registry = EnhancedModelRegistry(base_registry)
    model_identifier = model_registry.get_model_by_string(cfg.model_name)
    if model_identifier is None:
        raise ValueError(f"Unknown model: {cfg.model_name}")

    # Generation config
    generation_config = GenerationExperimentConfig(
        dataset_name=cfg.dataset.type.lower(),
        prompt_strategy=prompt_strategy_enum,
        generation_strategy=generation_strategy_enum,
        num_repeats=1,
        num_shots=0,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k_concentration=3,
    )

    # Pipeline config
    pipeline_config = IntegratedPipelineConfig(
        model_identifier=model_identifier,
        device=cfg.hardware.device,
        max_memory=max_memory,
        apply_compile=cfg.hardware.apply_compile,
        random_seed=42,
        dataset_config=dataset_config,
        num_excel_entries=10,
        generation_config=generation_config,
        batch_config=BatchConfig(batch_size=batch_size, shuffle=False, drop_last=False),
        pipeline_types=pipeline_types_enum,
        evaluation_config=EVALUATION_PIPELINES_DICT,
        exp_id=cfg.exp_id,
    )

    if cfg.use_wandb:
        wandb.init(
            project="llm-reliability-eval",
            name=f"{cfg.exp_id}-{cfg.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info(f"Running: exp={cfg.exp_id} model={cfg.model_name} dataset={cfg.dataset.type}")
    pipeline = IntegratedEvaluationPipeline()
    start = time.time()
    results = pipeline.run_evaluation(pipeline_config)
    elapsed = time.time() - start

    summary = post_process_results(results)
    summary["execution_time"] = elapsed
    if cfg.use_wandb:
        wandb.log(summary)
    logger.info(f"Done in {elapsed:.1f}s")

    try:
        cleanup_config = ModelConfig(
            identifier=model_identifier,
            device=cfg.hardware.device,
            max_memory=max_memory,
            apply_compile=cfg.hardware.apply_compile,
        )
        model_paths = base_registry.get_model_paths(model_identifier)
        cleanup_config.model_path = model_paths.model_path
        cleanup_config.tokenizer_path = model_paths.tokenizer_path
        ModelManager().cleanup_model_cache(cleanup_config)
    except Exception as e:
        logger.warning(f"Cache cleanup failed (non-fatal): {e}")

    return summary


if __name__ == "__main__":
    main()
