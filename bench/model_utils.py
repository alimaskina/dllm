"""Helpers for Fast-dLLM v2 block-size configuration."""

from __future__ import annotations


def default_small_block_size(bd_size: int, *, equal_to_block: bool = False) -> int:
    """Return small_block_size for diffusion decoding.

    equal_to_block=True: no sub-blocking (parallel over full bd_size).
    Otherwise official ratio: bd_size // 4.
    """
    if equal_to_block:
        return bd_size
    if bd_size % 4 != 0:
        raise ValueError(f"bd_size must be divisible by 4, got {bd_size}")
    return bd_size // 4


def configure_block_size(model, bd_size: int) -> None:
    """Sync runtime block size with model internals (attention masks etc.)."""
    if hasattr(model, "model") and hasattr(model.model, "bd_size"):
        model.model.bd_size = bd_size
    if hasattr(model, "config") and hasattr(model.config, "bd_size"):
        model.config.bd_size = bd_size
