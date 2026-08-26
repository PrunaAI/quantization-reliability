import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
from src.config import CACHE_ROOT
from src.model_loading.common.model_config import ModelConfig
from src.model_loading.common.model_enums import QuantizationMethod
from src.model_loading.common.stats import ModelStats
from src.model_loading.loader_factory import ModelLoaderFactory
from src.model_loading.registry.registry import ModelRegistry
from src.model_loading.common.tokenizer_utils import get_actual_tokenizer


logger = logging.getLogger("reliability_eval")


class ModelManager:
    """High-level manager for model loading and configuration"""
    
    def __init__(self, model_save_path: Optional[str] = None):
        self.registry = ModelRegistry(model_save_path=model_save_path)
    
    def _prepare_tokenizer(self, tokenizer):
        """Prepare tokenizer with common configurations"""
        # Check if this is a processor (VLM) rather than a tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            # For processors, configure the underlying tokenizer
            actual_tokenizer = tokenizer.tokenizer
            actual_tokenizer.pad_token_id = actual_tokenizer.eos_token_id
            actual_tokenizer.padding_side = "left"
            if actual_tokenizer.model_max_length > 1e6:
                logger.warning(f"Tokenizer model max length reduced from {actual_tokenizer.model_max_length} to 2048")
                actual_tokenizer.model_max_length = 2048
        else:
            # Standard tokenizer
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
            if tokenizer.model_max_length > 1e6:
                logger.warning(f"Tokenizer model max length reduced from {tokenizer.model_max_length} to 2048")
                tokenizer.model_max_length = 2048
        
        return tokenizer
    
    def _collect_model_stats(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> ModelStats:
        """Collect and return model statistics"""
        # Handle processor vs tokenizer
        actual_tokenizer = get_actual_tokenizer(tokenizer)
        
        return ModelStats(
            cache_dir=getattr(model, 'cache_dir', None),
            model_max_length=actual_tokenizer.model_max_length,
            dtype=getattr(model, 'dtype', None),
            device=getattr(model, 'device', None),
            num_parameters=getattr(model, 'num_parameters', 0),
            memory_footprint=getattr(model, 'memory_footprint', 0) / (1024 ** 3),
            vocab_size=actual_tokenizer.vocab_size,
            pad_token_id=actual_tokenizer.eos_token_id,
            special_tokens=actual_tokenizer.special_tokens_map
        )
    
    def _validate_model_cache(self, loader, config: ModelConfig) -> None:
        """Delete cache directories if corrupt, forcing a clean re-download.
        Detects two failure modes:
          1. .incomplete files — download was interrupted mid-blob
          2. Broken symlinks  — snapshot dir exists but referenced blobs are missing
        Only validates HF-format paths (containing 'models--') since custom save paths
        don't use HF's download mechanism.
        """
        for cache_path in loader.get_model_cache_paths(config):
            if not cache_path.exists():
                continue
            if "models--" not in cache_path.name:
                continue
            incomplete_files = list(cache_path.rglob("*.incomplete"))
            broken_symlinks = [p for p in cache_path.rglob("*") if p.is_symlink() and not p.resolve().exists()]
            if incomplete_files or broken_symlinks:
                logger.warning(
                    f"Corrupt cache detected at {cache_path} — "
                    f"{len(incomplete_files)} incomplete file(s), {len(broken_symlinks)} broken symlink(s). "
                    f"Deleting and re-downloading..."
                )
                self._safe_rmtree(cache_path, config)

    def _safe_rmtree(self, cache_path: Path, config: ModelConfig, context: str = "corrupt cache") -> None:
        """Delete a cache directory only after passing all safety checks."""
        ceph_cache_root = CACHE_ROOT
        model_name = config.model_path.split("/")[-1]
        if not str(cache_path).startswith(ceph_cache_root):
            logger.error(f"Safety check failed: path outside ceph cache: {cache_path}")
            return
        if not cache_path.is_dir():
            logger.error(f"Safety check failed: path is not a directory: {cache_path}")
            return
        if model_name.lower() not in cache_path.name.lower():
            logger.error(f"Safety check failed: model name '{model_name}' not found in path '{cache_path.name}'")
            return
        if cache_path.parent == Path(ceph_cache_root):
            logger.error(f"Safety check failed: refusing to delete a top-level cache directory: {cache_path}")
            return
        shutil.rmtree(cache_path)
        logger.info(f"Deleted {context}: {cache_path}")

    def load_model(self, config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer based on configuration"""
        try:
            logger.info(f"Loading model: {config.identifier}")

            paths = self.registry.get_model_paths(config.identifier)
            if not paths:
                raise ValueError(f"No paths found for model: {config.identifier}")

            if config.identifier.is_local and not os.path.exists(paths.model_path):
                raise OSError(f"Local model path does not exist: {paths.model_path}")

            loader = ModelLoaderFactory.create_loader(quant_method=config.identifier.quantization or QuantizationMethod.NONE)

            # Update config with paths from registry
            config.model_path = paths.model_path
            config.tokenizer_path = paths.tokenizer_path

            # Check for incomplete downloads before loading — forces re-download if corrupt
            self._validate_model_cache(loader, config)

            try:
                model, tokenizer = loader.load_model(config)
            except Exception as e:
                logger.warning(f"Model loading failed: {e}. Clearing cache and retrying once...")
                try:
                    self.cleanup_model_cache(config)
                except Exception as cleanup_err:
                    logger.warning(f"Cache cleanup failed (non-fatal): {cleanup_err}")
                model, tokenizer = loader.load_model(config)

            model.NAME = str(config.identifier)
            tokenizer = self._prepare_tokenizer(tokenizer)

            stats = self._collect_model_stats(model, tokenizer)
            self._log_model_info(str(config.identifier), stats)

            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model {config.identifier}: {str(e)}")
            raise
    
    def cleanup_model_cache(self, config: ModelConfig) -> None:
        """Delete all cache paths for a specific model after an experiment.
        Safe no-op for GPTQ and other persistent loaders (they return empty list)."""
        loader = ModelLoaderFactory.create_loader(quant_method=config.identifier.quantization or QuantizationMethod.NONE)
        cache_paths = loader.get_model_cache_paths(config)
        if not cache_paths:
            logger.info(f"Cache cleanup not configured for {config.identifier}, skipping")
            return
        for cache_path in cache_paths:
            if not cache_path.exists():
                logger.info(f"Cache path does not exist, nothing to clean: {cache_path}")
                continue
            logger.info(f"Cleaning up model cache: {cache_path}")
            self._safe_rmtree(cache_path, config, context="model cache (post-run cleanup)")
            logger.info(f"Successfully cleaned up cache: {cache_path}")

    def _log_model_info(self, model_name: str, stats: ModelStats):
        """Log model information and statistics"""
        logger.info(f"Successfully loaded model: {model_name}")
        logger.info("Model configuration:")
        logger.info(f"- Cache directory: {stats.cache_dir}")
        logger.info(f"- Model max length: {stats.model_max_length}")
        logger.info(f"- Model dtype: {stats.dtype}")
        logger.info(f"- Model device: {stats.device}")
        logger.info(f"- Model parameters: {stats.num_parameters}")
        logger.info(f"- Memory footprint: {stats.memory_footprint:.2f} GB")
        logger.info(f"- Vocabulary size: {stats.vocab_size}")
        logger.info(f"- Padding token ID: {stats.pad_token_id}")
        logger.info(f"- Special tokens: {stats.special_tokens}")
