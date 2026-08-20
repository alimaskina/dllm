"""Dual-precision old KV cache storage."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache

from quantization import QuantizedTensor, apply_precision, quantize_dequantize


class DualPrecisionCache:
    """Hold FP16 canonical old cache plus optional selector/exec quant copies.

    Canonical ``fp16_cache`` is the full old DynamicCache (prompt + finished blocks).
    ``selector_view`` and ``exec_view`` are dequantized tensors per layer for
    controlled ablations (e.g. FP16 for selection, INT2 for sparse execution).
    """

    def __init__(self) -> None:
        self.fp16_cache: DynamicCache | None = None
        self.cache_len: int = 0
        # layer_id -> dequantized K/V for selector precision
        self.selector_k: dict[int, torch.Tensor] = {}
        self.selector_v: dict[int, torch.Tensor] = {}
        # layer_id -> dequantized K/V for execution precision (full old cache)
        self.exec_k: dict[int, torch.Tensor] = {}
        self.exec_v: dict[int, torch.Tensor] = {}
        # layer_id -> sparse exec dequantized (only selected indices)
        self.sparse_exec_k: dict[int, torch.Tensor] = {}
        self.sparse_exec_v: dict[int, torch.Tensor] = {}

    @staticmethod
    def clone_dynamic_cache(past_key_values: DynamicCache) -> DynamicCache:
        from transformers.cache_utils import DynamicCache

        cloned = DynamicCache()
        for layer_id in range(len(past_key_values)):
            cloned.key_cache.append(past_key_values.key_cache[layer_id].clone())
            cloned.value_cache.append(past_key_values.value_cache[layer_id].clone())
        return cloned

    def bind_fp16(self, past_key_values: DynamicCache | None) -> None:
        if past_key_values is None:
            self.fp16_cache = None
            self.cache_len = 0
            return
        self.fp16_cache = past_key_values
        self.cache_len = past_key_values.get_seq_length()

    def build_precision_views(
        self,
        *,
        selector_k_bits: int | str,
        selector_v_bits: int | str,
        exec_k_bits: int | str,
        exec_v_bits: int | str,
        k_group_size: int = 1,
        keys_pre_rope: bool = False,
        model=None,
    ) -> None:
        """Populate selector/exec dequant views from canonical FP16 cache."""
        if self.fp16_cache is None or self.cache_len == 0:
            return

        cos = sin = None
        if keys_pre_rope:
            if model is None:
                raise ValueError("model required for keys_pre_rope")
            import sys
            from pathlib import Path

            _kvq = Path(__file__).resolve().parent.parent / "kv_quant"
            if str(_kvq) not in sys.path:
                sys.path.insert(0, str(_kvq))
            from kv_cache_quant import apply_rope_keys, inverse_rope_keys, rope_cos_sin

            key0 = self.fp16_cache.key_cache[0]
            cos, sin = rope_cos_sin(
                model, self.cache_len, key0.shape[0], key0.device, key0.dtype
            )

        self.selector_k.clear()
        self.selector_v.clear()
        self.exec_k.clear()
        self.exec_v.clear()

        for layer_id in range(len(self.fp16_cache)):
            k = self.fp16_cache.key_cache[layer_id][..., : self.cache_len, :].clone()
            v = self.fp16_cache.value_cache[layer_id][..., : self.cache_len, :].clone()

            if keys_pre_rope and cos is not None:
                k_nr = inverse_rope_keys(k, cos, sin)
                k_sel = apply_precision(k_nr, selector_k_bits, "k_per_channel", group_size=k_group_size)
                k_sel = apply_rope_keys(k_sel, cos, sin)
                k_exec = apply_precision(k_nr, exec_k_bits, "k_per_channel", group_size=k_group_size)
                k_exec = apply_rope_keys(k_exec, cos, sin)
            else:
                k_sel = apply_precision(k, selector_k_bits, "k_per_channel", group_size=k_group_size)
                k_exec = apply_precision(k, exec_k_bits, "k_per_channel", group_size=k_group_size)

            self.selector_k[layer_id] = k_sel
            self.selector_v[layer_id] = apply_precision(
                v, selector_v_bits, "v_per_token", group_size=k_group_size
            )
            self.exec_k[layer_id] = k_exec
            self.exec_v[layer_id] = apply_precision(
                v, exec_v_bits, "v_per_token", group_size=k_group_size
            )

    def build_sparse_exec(
        self,
        per_layer_indices: dict[int, list[int]],
    ) -> None:
        """Extract selected old-cache tokens into sparse exec views."""
        self.sparse_exec_k.clear()
        self.sparse_exec_v.clear()
        for layer_id, indices in per_layer_indices.items():
            if not indices:
                continue
            idx = torch.tensor(indices, device=self.exec_k[layer_id].device, dtype=torch.long)
            self.sparse_exec_k[layer_id] = self.exec_k[layer_id].index_select(2, idx)
            self.sparse_exec_v[layer_id] = self.exec_v[layer_id].index_select(2, idx)

    def get_layer_old_kv(
        self,
        layer_id: int,
        *,
        phase: str,
        sparse: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return old-cache K/V dequant view (always full length for mask compat)."""
        if self.cache_len == 0:
            return None, None
        if phase == "selector":
            return self.selector_k.get(layer_id), self.selector_v.get(layer_id)
        return self.exec_k.get(layer_id), self.exec_v.get(layer_id)
