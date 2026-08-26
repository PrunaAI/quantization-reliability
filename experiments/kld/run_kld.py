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

import json
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra

torch.hub.set_dir(HF_HOME)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src

from src.model_loading.registry.enhanced_registry import EnhancedModelRegistry
from src.model_loading.registry.registry import ModelRegistry
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.manager import ModelManager
from src.loggers.setup_logging import setup_logging

logger = setup_logging()

base_registry = ModelRegistry()
model_registry = EnhancedModelRegistry(base_registry)


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


def parse_model_name(model_name: str):
    """Extract base model, quant method, and bit-width from a model name string."""
    quant_methods = {"hqq", "bnb", "gptq", "awq", "quanto"}
    parts = model_name.split("_")
    base_parts, quant_method, bits = [], None, None
    for part in parts:
        if part in quant_methods:
            quant_method = part
        elif "bit" in part:
            bits = int(part.replace("bit", ""))
        elif quant_method is None:
            base_parts.append(part)
    if quant_method is None:
        quant_method, bits = "fp16", 16
    return "_".join(base_parts), quant_method, bits


def load_model(model_name: str, max_memory: Optional[dict], apply_compile: bool, device: str):
    model_identifier = model_registry.get_model_by_string(model_name)
    model_paths = base_registry.get_model_paths(model_identifier)

    config = ModelConfig(
        identifier=model_identifier,
        device=device,
        trust_remote_code=True,
        max_memory=max_memory,
        apply_compile=apply_compile,
        cache_dir=HF_HOME,
    )
    config.model_path = model_paths.model_path
    config.tokenizer_path = model_paths.tokenizer_path

    name_lower = model_name.lower()
    if "hqq" in name_lower:
        from src.model_loading.loaders.hqq import HQQModelLoader
        loader = HQQModelLoader()
    elif "bnb" in name_lower:
        from src.model_loading.loaders.bitsandbytes import BitsAndBytesModelLoader
        loader = BitsAndBytesModelLoader()
    elif "quanto" in name_lower:
        from src.model_loading.loaders.quanto import QuantoModelLoader
        loader = QuantoModelLoader()
    elif "awq" in name_lower:
        from src.model_loading.loaders.awq import AWQModelLoader
        loader = AWQModelLoader()
    elif "gptq" in name_lower:
        from src.model_loading.loaders.gptq import GPTQModelLoader
        loader = GPTQModelLoader()
    else:
        from src.model_loading.loaders.standard import StandardModelLoader
        loader = StandardModelLoader()

    model, tokenizer = loader.load_model(config)

    # GPTQ leaves non-quantized params on CPU — move float params to target device
    if "gptq" in name_lower:
        _float_dtypes = (torch.float16, torch.float32, torch.bfloat16)
        for p in model.parameters():
            if p.dtype in _float_dtypes:
                p.data = p.data.to(device)
        for b in model.buffers():
            if b.dtype in _float_dtypes:
                b.data = b.data.to(device)

    return model, tokenizer


def extract_hidden_states(model, tokenizer, text: str, device: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [h.cpu() for h in outputs.hidden_states]


def compute_errors(fp_hidden, quant_hidden):
    per_layer = [torch.norm(fp - q, p=2).item() for fp, q in zip(fp_hidden, quant_hidden)]
    return per_layer, list(np.cumsum(per_layer))


def _load_raw_texts(n_samples: int, dataset: str) -> list:
    if dataset == "c4":
        raw = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = [s["text"] for s in raw if len(s["text"]) > 50][:n_samples]
    else:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in data["text"] if len(t) > 50][:n_samples]
    return texts


def compute_kl_divergence(ref_model, eval_model, tokenizer, device: str, seqlen: int, n_samples: int, dataset: str):
    text = "\n\n".join(_load_raw_texts(n_samples, dataset))
    data = tokenizer(text, return_tensors="pt").input_ids.to(device)
    n_seqs = data.numel() // seqlen

    kls, kls_rev, jss = [], [], []
    with tqdm(range(n_seqs), desc="Computing KL divergence") as bar:
        for i in bar:
            batch = data[:, i * seqlen:(i + 1) * seqlen]
            with torch.no_grad():
                ref_logits = ref_model(batch).logits
                eval_logits = eval_model(batch).logits

            ref_probs = torch.softmax(ref_logits, dim=-1)
            eval_probs = torch.softmax(eval_logits, dim=-1)
            ref_log = torch.log_softmax(ref_logits, dim=-1)
            eval_log = torch.log_softmax(eval_logits, dim=-1)

            kls.append(torch.sum(ref_probs * (ref_log - eval_log), dim=-1).mean(-1).item())
            kls_rev.append(torch.sum(eval_probs * (eval_log - ref_log), dim=-1).mean(-1).item())

            mean_probs = 0.5 * (ref_probs + eval_probs)
            js = 0.5 * (F.kl_div(mean_probs.log(), ref_probs, reduction="none").sum(-1)
                        + F.kl_div(mean_probs.log(), eval_probs, reduction="none").sum(-1))
            jss.append((1 - torch.sqrt(torch.clamp(js, min=1e-10))).mean(-1).item())

            bar.set_description(f"KL(p||q): {np.mean(kls):.4f}")

    return {
        "token_kl_pq": (np.mean(kls), np.std(kls)),
        "token_kl_qp": (np.mean(kls_rev), np.std(kls_rev)),
        "token_js_sim": (np.mean(jss), np.std(jss)),
    }


def save_results(results: dict, exp_id: str, base_model: str, quant_method: str, bits: int, output_dir: Path):
    model_dir = output_dir / exp_id / base_model / quant_method
    model_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "layer_id": range(len(results["mean_per_layer"])),
        "mean_per_layer": results["mean_per_layer"],
        "mean_accumulated": results["mean_accumulated"],
    }).to_csv(model_dir / f"{bits}bit_hidden_states.csv", index=False)

    kl = results["kl_metrics"]
    with open(model_dir / f"{bits}bit_kl_metrics.json", "w") as f:
        json.dump({
            "token_kl_pq_mean": kl["token_kl_pq"][0], "token_kl_pq_std": kl["token_kl_pq"][1],
            "token_kl_qp_mean": kl["token_kl_qp"][0], "token_kl_qp_std": kl["token_kl_qp"][1],
            "token_js_sim_mean": kl["token_js_sim"][0], "token_js_sim_std": kl["token_js_sim"][1],
        }, f, indent=2)

    logger.info(f"Saved results to {model_dir}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict:
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    authenticate_huggingface()
    set_seeds(cfg.random_seed)

    max_memory = OmegaConf.to_container(cfg.hardware.max_memory) if cfg.hardware.max_memory else None
    if max_memory:
        max_memory = {int(k): v for k, v in max_memory.items()}

    base_model, quant_method, bits = parse_model_name(cfg.model_name)
    device = cfg.hardware.device
    logger.info(f"model={cfg.model_name}  base={base_model}  quant={quant_method}  bits={bits}")

    sample_texts = _load_raw_texts(cfg.num_samples, cfg.dataset.name)

    # Load full-precision model
    fp_model, fp_tokenizer = load_model(base_model, max_memory, cfg.apply_compile, device)

    # Extract FP hidden states
    fp_hidden_all = []
    for i, text in enumerate(sample_texts):
        logger.info(f"FP hidden states {i+1}/{len(sample_texts)}")
        fp_hidden_all.append(extract_hidden_states(fp_model, fp_tokenizer, text, device))

    # Load quantized model (skip if already fp)
    if quant_method == "fp16":
        quant_model, quant_tokenizer = fp_model, fp_tokenizer
    else:
        quant_model, quant_tokenizer = load_model(cfg.model_name, max_memory, cfg.apply_compile, device)

    # Compute hidden state errors
    all_per_layer, all_accumulated = [], []
    for i, text in enumerate(sample_texts):
        logger.info(f"Quant hidden states {i+1}/{len(sample_texts)}")
        q_hidden = extract_hidden_states(quant_model, fp_tokenizer, text, device)
        per_layer, accumulated = compute_errors(fp_hidden_all[i], q_hidden)
        all_per_layer.append(per_layer)
        all_accumulated.append(accumulated)

    mean_per_layer = np.array(all_per_layer).mean(axis=0)
    mean_accumulated = np.array(all_accumulated).mean(axis=0)

    # Compute KL divergence
    kl_metrics = compute_kl_divergence(
        fp_model, quant_model, fp_tokenizer, device,
        seqlen=cfg.seqlen, n_samples=cfg.num_samples, dataset=cfg.dataset.name,
    )

    results = {"mean_per_layer": mean_per_layer, "mean_accumulated": mean_accumulated, "kl_metrics": kl_metrics}
    save_results(results, cfg.exp_id, base_model, quant_method, bits, Path(cfg.output_dir))

    # Save per-run CSV row
    row = {
        "_id": str(uuid.uuid4()),
        "exp_id": cfg.exp_id,
        "model_name": cfg.model_name,
        "base_model": base_model,
        "quant_method": quant_method,
        "bits": bits,
        "dataset": cfg.dataset.name.upper(),
        "kl_pq_mean": kl_metrics["token_kl_pq"][0],
        "kl_pq_std": kl_metrics["token_kl_pq"][1],
        "kl_qp_mean": kl_metrics["token_kl_qp"][0],
        "kl_qp_std": kl_metrics["token_kl_qp"][1],
        "js_sim_mean": kl_metrics["token_js_sim"][0],
        "js_sim_std": kl_metrics["token_js_sim"][1],
    }
    out_path = Path(cfg.results_dir) / cfg.exp_id
    out_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path / f"kl_divergence_{cfg.model_name}.csv", index=False)

    if cfg.use_wandb:
        import wandb
        wandb.init(
            project="llm-kld-eval",
            name=f"{cfg.exp_id}-{cfg.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float))})

    # Cleanup
    del fp_model
    if quant_method != "fp16":
        del quant_model
    torch.cuda.empty_cache()

    try:
        mgr = ModelManager()
        for name in ([base_model] + ([cfg.model_name] if quant_method != "fp16" else [])):
            ident = model_registry.get_model_by_string(name)
            paths = base_registry.get_model_paths(ident)
            mc = ModelConfig(identifier=ident, device=device, apply_compile=cfg.apply_compile)
            mc.model_path = paths.model_path
            mc.tokenizer_path = paths.tokenizer_path
            mgr.cleanup_model_cache(mc)
    except Exception as e:
        logger.warning(f"Cache cleanup failed (non-fatal): {e}")

    summary = {**row, "kl_pq_mean": kl_metrics["token_kl_pq"][0], "js_sim_mean": kl_metrics["token_js_sim"][0]}
    logger.info(f"KL(p||q): {kl_metrics['token_kl_pq'][0]:.4f} ± {kl_metrics['token_kl_pq'][1]:.4f}")
    logger.info(f"JS Sim:   {kl_metrics['token_js_sim'][0]:.4f} ± {kl_metrics['token_js_sim'][1]:.4f}")
    return summary


if __name__ == "__main__":
    main()
