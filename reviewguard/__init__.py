"""ReviewGuard AI backend. CPU-only inference with project-local caches."""

import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / "models/hf_cache"
os.environ.setdefault("HF_HOME", str(_cache))
os.environ.setdefault("HF_XET_CACHE", str(_cache / "xet"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
__version__ = "1.0.0"

os.environ.setdefault("MPLCONFIGDIR", str(_cache.parent / "mpl_cache"))
