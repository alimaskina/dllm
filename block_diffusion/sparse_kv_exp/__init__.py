"""Experimental sparse old-cache + low-bit KV/Q attention for Fast-dLLM-v2."""

from .config import ExperimentConfig, load_config
from .generation import batch_sample_sparse_kv

__all__ = ["ExperimentConfig", "load_config", "batch_sample_sparse_kv"]
