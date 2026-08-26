import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_HOME = Path.home() / ".cache" / "quantization-reliability"

DATA_DIR = os.environ.get("QR_DATA_DIR", str(REPO_ROOT / "data"))
MODEL_SAVE_DIR = os.environ.get("QR_MODEL_SAVE_DIR", str(_DEFAULT_HOME / "models"))
MODEL_CACHE_ROOT = os.environ.get("QR_MODEL_CACHE_ROOT", str(_DEFAULT_HOME))
CACHE_ROOT = os.environ.get("QR_CACHE_ROOT", str(_DEFAULT_HOME / ".cache"))
HF_CACHE_ROOT = os.environ.get("QR_HF_CACHE_ROOT", f"{CACHE_ROOT}/huggingface/hub")
RESULTS_DIR = os.environ.get("QR_RESULTS_DIR", str(REPO_ROOT / "results"))
