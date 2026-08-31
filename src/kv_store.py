"""Dual-precision old KV cache storage."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache

from quantization import apply_precision

_KVQ = Path(__file__).resolve().parent.parent / "kv_quant"
if str(_KVQ) not in sys.path:
    sys.path.insert(0, str(_KVQ))
from kv_cache_quant import quantize_key_roundtrip, quantize_value_roundtrip  # noqa: E402


def _apply_key_precision(
    k: torch.Tensor,
    bits: int | str,
    *,
    kivi_group_size: int,
    kivi_residual_length: int,
    k_quant_scheme: str = "kivi",
) -> torch.Tensor:
    """Quantize keys: KIVI per-channel, naive per-token, or Hadamard per-token."""
    b = str(bits).lower()
    if b in ("fp16", "bf16", "16", "float16", "bfloat16") or int(bits) >= 16:
        return k.clone()
    return quantize_key_roundtrip(
        k,
        int(bits),
        scheme=k_quant_scheme,
        group_size=kivi_group_size,
        residual_length=kivi_residual_length,
    )


def _apply_value_precision(
    v: torch.Tensor,
    bits: int | str,
    *,
    v_quant_scheme: str = "kivi",
) -> torch.Tensor:
    """Quantize values: KIVI per-token or Hadamard per-token. All tokens (no residual)."""
    b = str(bits).lower()
    if b in ("fp16", "bf16", "16", "float16", "bfloat16") or int(bits) >= 16:
        return v.clone()
    if v_quant_scheme == "kivi":
        return apply_precision(v, bits, "v_per_token")
    return quantize_value_roundtrip(
        v, int(bits), scheme=v_quant_scheme, residual_length=0
    )


def _get_cache_kv(cache, layer_id):
    """Compat shim: transformers <5 uses `key_cache/value_cache` lists;
    transformers >=5 uses `layers[i].keys/values`.
    """
    if hasattr(cache, "key_cache") and layer_id < len(getattr(cache, "key_cache", [])):
        return cache.key_cache[layer_id], cache.value_cache[layer_id]
    return cache.layers[layer_id].keys, cache.layers[layer_id].values


def _cache_num_layers(cache) -> int:
    if hasattr(cache, "key_cache") and len(getattr(cache, "key_cache", [])) > 0:
        return len(cache.key_cache)
    return len(cache.layers)


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
        kivi_group_size: int = 32,
        kivi_residual_length: int = 32,
        keys_pre_rope: bool = False,
        k_quant_scheme: str = "kivi",
        v_quant_scheme: str = "kivi",
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

        for layer_id in range(_cache_num_layers(self.fp16_cache)):
            k_full, v_full = _get_cache_kv(self.fp16_cache, layer_id)
            k = k_full[..., : self.cache_len, :].clone()
            v = v_full[..., : self.cache_len, :].clone()

            if keys_pre_rope and cos is not None:
                k_nr = inverse_rope_keys(k, cos, sin)
                k_sel = _apply_key_precision(
                    k_nr, selector_k_bits,
                    kivi_group_size=kivi_group_size,
                    kivi_residual_length=kivi_residual_length,
                    k_quant_scheme=k_quant_scheme,
                )
                k_sel = apply_rope_keys(k_sel, cos, sin)
                k_exec = _apply_key_precision(
                    k_nr, exec_k_bits,
                    kivi_group_size=kivi_group_size,
                    kivi_residual_length=kivi_residual_length,
                    k_quant_scheme=k_quant_scheme,
                )
                k_exec = apply_rope_keys(k_exec, cos, sin)
            else:
                k_sel = _apply_key_precision(
                    k, selector_k_bits,
                    kivi_group_size=kivi_group_size,
                    kivi_residual_length=kivi_residual_length,
                    k_quant_scheme=k_quant_scheme,
                )
                k_exec = _apply_key_precision(
                    k, exec_k_bits,
                    kivi_group_size=kivi_group_size,
                    kivi_residual_length=kivi_residual_length,
                    k_quant_scheme=k_quant_scheme,
                )

            self.selector_k[layer_id] = k_sel
            self.selector_v[layer_id] = _apply_value_precision(
                v, selector_v_bits, v_quant_scheme=v_quant_scheme
            )
            self.exec_k[layer_id] = k_exec
            self.exec_v[layer_id] = _apply_value_precision(
                v, exec_v_bits, v_quant_scheme=v_quant_scheme
            )

    def build_sparse_exec(
        self,
        per_layer_indices: dict[int, list[int]],
        *,
        requantize: bool = False,
        requantize_k: bool | None = None,
        requantize_v: bool | None = None,
        source: str = "exec",
        k_bits: int | str = "fp16",
        v_bits: int | str = "fp16",
        kivi_group_size: int = 32,
        kivi_residual_length: int = 32,
        k_quant_scheme: str = "kivi",
        v_quant_scheme: str = "kivi",
    ) -> None:
        """Build sparse exec overrides for old-cache K/V.

        Note: SDPA hook keeps full key length for mask compatibility, so the
        stored tensors here keep the same sequence length as ``exec_k/exec_v``.
        Only the *selected* columns are overwritten (optionally requantized).
        """
        if source not in ("exec", "selector"):
            raise ValueError(f"source must be 'exec' or 'selector', got {source!r}")
        rq_k = requantize if requantize_k is None else bool(requantize_k)
        rq_v = requantize if requantize_v is None else bool(requantize_v)
        self.sparse_exec_k.clear()
        self.sparse_exec_v.clear()
        if (not rq_k and not rq_v) and source == "exec":
            # Nothing to do: exec will read ``exec_k/exec_v`` directly.
            return
        for layer_id, indices in per_layer_indices.items():
            if not indices:
                continue
            # Build full-length exec override.
            base_full_k = self.exec_k[layer_id]
            base_full_v = self.exec_v[layer_id]
            override_k = base_full_k.clone()
            override_v = base_full_v.clone()

            src_k = base_full_k if source == "exec" else self.selector_k[layer_id]
            src_v = base_full_v if source == "exec" else self.selector_v[layer_id]
            idx = torch.tensor(indices, device=src_k.device, dtype=torch.long)
            k_sel = src_k.index_select(2, idx)
            v_sel = src_v.index_select(2, idx)
            if rq_k:
                k_sel = _apply_key_precision(
                    k_sel,
                    k_bits,
                    kivi_group_size=kivi_group_size,
                    kivi_residual_length=kivi_residual_length,
                    k_quant_scheme=k_quant_scheme,
                )
            if rq_v:
                v_sel = _apply_value_precision(v_sel, v_bits, v_quant_scheme=v_quant_scheme)

            override_k.index_copy_(2, idx, k_sel)
            override_v.index_copy_(2, idx, v_sel)
            self.sparse_exec_k[layer_id] = override_k
            self.sparse_exec_v[layer_id] = override_v

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
        if sparse and layer_id in self.sparse_exec_k and layer_id in self.sparse_exec_v:
            return self.sparse_exec_k.get(layer_id), self.sparse_exec_v.get(layer_id)
        return self.exec_k.get(layer_id), self.exec_v.get(layer_id)
