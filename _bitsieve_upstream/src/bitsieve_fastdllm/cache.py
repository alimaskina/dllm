from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import QuantizationConfig
from .reference import (
    PackedKeys,
    PackedValues,
    dequantize_keys,
    dequantize_values,
    gather_per_kv_head,
    values_per_byte,
)


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


@dataclass(slots=True)
class AppendPlan:
    old_length: int
    new_tokens: int
    old_residual: int
    quantize_from_residual: int
    new_residual: int
    staged_layers: set[int]


@dataclass(slots=True)
class LayerCacheView:
    k_bits: int
    v_bits: int
    length: int
    quantized_length: int
    residual_length: int
    key_token_group: int
    value_channel_group: int
    k_q: torch.Tensor | None
    k_scale: torch.Tensor | None
    k_zero: torch.Tensor | None
    v_q: torch.Tensor | None
    v_scale: torch.Tensor | None
    v_zero: torch.Tensor | None
    k_fp: torch.Tensor | None
    v_fp: torch.Tensor | None
    k_residual: torch.Tensor | None
    v_residual: torch.Tensor | None


    head_dim: int | None = None


class PackedKVCache:

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_tokens: int,
        quant: QuantizationConfig,
        device: torch.device | str,
        compute_dtype: torch.dtype = torch.bfloat16,
        backend: str = "auto",
    ) -> None:
        quant.validate(head_dim=head_dim)
        if num_layers <= 0 or batch_size <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("all cache dimensions must be positive")
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.max_tokens = int(max_tokens)
        self.quant = quant
        self.device = torch.device(device)
        self.compute_dtype = compute_dtype
        self.param_dtype = _dtype_from_name(quant.param_dtype)
        if backend not in ("auto", "triton", "torch"):
            raise ValueError(f"unknown cache backend: {backend}")
        self.backend = backend

        self.length = 0
        self.quantized_length = 0
        self.residual_length = 0
        self._append_plan: AppendPlan | None = None

        l, b, h, d, n = (
            self.num_layers,
            self.batch_size,
            self.num_kv_heads,
            self.head_dim,
            self.max_tokens,
        )
        gk = quant.key_token_group
        gv = quant.value_channel_group
        residual_capacity = quant.residual_tokens + gk - 1
        self.residual_capacity = max(1, residual_capacity)

        self.k_fp: torch.Tensor | None = None
        self.v_fp: torch.Tensor | None = None
        self.k_q: torch.Tensor | None = None
        self.k_scale: torch.Tensor | None = None
        self.k_zero: torch.Tensor | None = None
        self.v_q: torch.Tensor | None = None
        self.v_scale: torch.Tensor | None = None
        self.v_zero: torch.Tensor | None = None
        self.k_residual: torch.Tensor | None = None
        self.v_residual: torch.Tensor | None = None

        if quant.k_bits == 16:
            self.k_fp = torch.empty((l, b, h, n, d), device=self.device, dtype=compute_dtype)
        else:
            k_vpb = values_per_byte(quant.k_bits)
            max_groups = math.ceil(n / gk)
            self.k_q = torch.empty(
                (l, b, h, max_groups, d, gk // k_vpb),
                device=self.device,
                dtype=torch.uint8,
            )
            self.k_scale = torch.empty(
                (l, b, h, max_groups, d), device=self.device, dtype=self.param_dtype
            )
            self.k_zero = torch.empty_like(self.k_scale)
            self.k_residual = torch.empty(
                (l, b, h, self.residual_capacity, d),
                device=self.device,
                dtype=compute_dtype,
            )

        if quant.v_bits == 16:
            self.v_fp = torch.empty((l, b, h, n, d), device=self.device, dtype=compute_dtype)
        else:
            v_vpb = values_per_byte(quant.v_bits)
            value_groups = d // gv
            self.v_q = torch.empty(
                (l, b, h, n, value_groups, gv // v_vpb),
                device=self.device,
                dtype=torch.uint8,
            )
            self.v_scale = torch.empty(
                (l, b, h, n, value_groups), device=self.device, dtype=self.param_dtype
            )
            self.v_zero = torch.empty_like(self.v_scale)
            self.v_residual = torch.empty(
                (l, b, h, self.residual_capacity, d),
                device=self.device,
                dtype=compute_dtype,
            )

    def __len__(self) -> int:


        return self.num_layers if self.length > 0 else 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def get_max_cache_shape(self) -> int:
        return self.max_tokens

    def reset(self) -> None:
        self.length = 0
        self.quantized_length = 0
        self.residual_length = 0
        self._append_plan = None

    @property
    def append_in_progress(self) -> bool:
        return self._append_plan is not None

    def begin_append(self, new_tokens: int) -> AppendPlan:
        if self._append_plan is not None:
            raise RuntimeError("another cache append is already in progress")
        if new_tokens <= 0:
            raise ValueError("new_tokens must be positive")
        if self.length + new_tokens > self.max_tokens:
            raise RuntimeError(
                f"KV cache capacity exceeded: {self.length}+{new_tokens}>{self.max_tokens}"
            )

        if self.quant.k_bits == 16 and self.quant.v_bits == 16:
            move = 0
            new_residual = 0
        else:
            total_tail = self.residual_length + new_tokens
            excess = max(0, total_tail - self.quant.residual_tokens)
            move = (excess // self.quant.key_token_group) * self.quant.key_token_group
            new_residual = total_tail - move
            if new_residual >= self.residual_capacity:
                raise AssertionError("residual capacity calculation is incorrect")

        plan = AppendPlan(
            old_length=self.length,
            new_tokens=int(new_tokens),
            old_residual=self.residual_length,
            quantize_from_residual=move,
            new_residual=new_residual,
            staged_layers=set(),
        )
        self._append_plan = plan
        return plan

    def abort_append(self) -> None:
        self._append_plan = None

    def stage_layer(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        plan = self._append_plan
        if plan is None:
            raise RuntimeError("begin_append must be called before stage_layer")
        if layer_idx in plan.staged_layers:
            raise RuntimeError(f"layer {layer_idx} was staged twice")
        expected = (self.batch_size, self.num_kv_heads, plan.new_tokens, self.head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(
                f"append tensor shape mismatch: expected {expected}, got K={tuple(key.shape)}, "
                f"V={tuple(value.shape)}"
            )
        key = key.to(self.compute_dtype)
        value = value.to(self.compute_dtype)

        start = plan.old_length
        end = start + plan.new_tokens
        if self.quant.k_bits == 16:
            assert self.k_fp is not None
            self.k_fp[layer_idx, :, :, start:end, :].copy_(key)
        if self.quant.v_bits == 16:
            assert self.v_fp is not None
            self.v_fp[layer_idx, :, :, start:end, :].copy_(value)

        if self.quant.k_bits < 16 or self.quant.v_bits < 16:
            old_r = plan.old_residual
            if old_r:
                k_parts = []
                v_parts = []
                if self.quant.k_bits < 16:
                    assert self.k_residual is not None
                    k_parts.append(self.k_residual[layer_idx, :, :, :old_r, :])
                if self.quant.v_bits < 16:
                    assert self.v_residual is not None
                    v_parts.append(self.v_residual[layer_idx, :, :, :old_r, :])
            else:
                k_parts = []
                v_parts = []

            if self.quant.k_bits < 16:
                k_parts.append(key)
                k_tail = k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=2)
            else:
                k_tail = None
            if self.quant.v_bits < 16:
                v_parts.append(value)
                v_tail = v_parts[0] if len(v_parts) == 1 else torch.cat(v_parts, dim=2)
            else:
                v_tail = None

            move = plan.quantize_from_residual
            if move:
                group_start = self.quantized_length // self.quant.key_token_group
                group_count = move // self.quant.key_token_group
                if k_tail is not None:
                    assert self.k_q is not None and self.k_scale is not None and self.k_zero is not None
                    q_out = self.k_q[
                        layer_idx, :, :, group_start : group_start + group_count, :, :
                    ]
                    s_out = self.k_scale[
                        layer_idx, :, :, group_start : group_start + group_count, :
                    ]
                    z_out = self.k_zero[
                        layer_idx, :, :, group_start : group_start + group_count, :
                    ]
                    from .kernels.ops import quantize_key_groups_into

                    quantize_key_groups_into(
                        k_tail[:, :, :move, :].contiguous(),
                        q_out,
                        s_out,
                        z_out,
                        bits=self.quant.k_bits,
                        token_group=self.quant.key_token_group,
                        backend=self.backend,
                    )
                if v_tail is not None:
                    assert self.v_q is not None and self.v_scale is not None and self.v_zero is not None
                    v_start = self.quantized_length
                    v_end = v_start + move
                    qv_out = self.v_q[layer_idx, :, :, v_start:v_end, :, :]
                    sv_out = self.v_scale[layer_idx, :, :, v_start:v_end, :]
                    zv_out = self.v_zero[layer_idx, :, :, v_start:v_end, :]
                    from .kernels.ops import quantize_values_into

                    quantize_values_into(
                        v_tail[:, :, :move, :].contiguous(),
                        qv_out,
                        sv_out,
                        zv_out,
                        bits=self.quant.v_bits,
                        channel_group=self.quant.value_channel_group,
                        backend=self.backend,
                    )

            remain = plan.new_residual
            if k_tail is not None:
                assert self.k_residual is not None
                self.k_residual[layer_idx, :, :, :remain, :].copy_(k_tail[:, :, move:, :])
            if v_tail is not None:
                assert self.v_residual is not None
                self.v_residual[layer_idx, :, :, :remain, :].copy_(v_tail[:, :, move:, :])

        plan.staged_layers.add(layer_idx)

    def commit_append(self) -> None:
        plan = self._append_plan
        if plan is None:
            raise RuntimeError("no append is in progress")
        expected = set(range(self.num_layers))
        if plan.staged_layers != expected:
            missing = sorted(expected - plan.staged_layers)
            raise RuntimeError(f"cannot commit: layers not staged: {missing}")
        self.length += plan.new_tokens
        if self.quant.k_bits < 16 or self.quant.v_bits < 16:
            self.quantized_length += plan.quantize_from_residual
            self.residual_length = plan.new_residual
        self._append_plan = None
        if self.quantized_length + self.residual_length != self.length and (
            self.quant.k_bits < 16 or self.quant.v_bits < 16
        ):
            raise AssertionError("cache length invariant failed")

    def load_dynamic_cache(self, dynamic_cache) -> None:
        if self.length != 0 or self._append_plan is not None:
            raise RuntimeError("load_dynamic_cache requires an empty cache")
        if dynamic_cache is None:
            return
        length = int(dynamic_cache.get_seq_length())
        if length == 0:
            return
        if length > self.max_tokens:
            raise RuntimeError(f"prefill cache length {length} exceeds capacity {self.max_tokens}")
        self.begin_append(length)
        try:
            for layer_idx in range(self.num_layers):
                # Three cache APIs across the transformers versions this has
                # to serve: tuple indexing, the key_cache/value_cache lists,
                # and (5.x) layers[i].keys / .values.
                key = value = None
                try:
                    key, value = dynamic_cache[layer_idx]
                except Exception:
                    if hasattr(dynamic_cache, "key_cache"):
                        key = dynamic_cache.key_cache[layer_idx]
                        value = dynamic_cache.value_cache[layer_idx]
                    else:
                        layer = dynamic_cache.layers[layer_idx]
                        key, value = layer.keys, layer.values
                if key is None or value is None:
                    raise TypeError(
                        f"cannot read layer {layer_idx} out of {type(dynamic_cache).__name__}"
                    )
                self.stage_layer(
                    layer_idx,
                    key[..., :length, :].contiguous(),
                    value[..., :length, :].contiguous(),
                )
            self.commit_append()
        except Exception:
            self.abort_append()
            raise

    def layer_view(self, layer_idx: int) -> LayerCacheView:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(layer_idx)
        qgroups = self.quantized_length // self.quant.key_token_group
        return LayerCacheView(
            k_bits=self.quant.k_bits,
            v_bits=self.quant.v_bits,
            length=self.length,
            quantized_length=self.quantized_length,
            residual_length=self.residual_length,
            key_token_group=self.quant.key_token_group,
            value_channel_group=self.quant.value_channel_group,
            k_q=None if self.k_q is None else self.k_q[layer_idx, :, :, :qgroups, :, :],
            k_scale=(
                None if self.k_scale is None else self.k_scale[layer_idx, :, :, :qgroups, :]
            ),
            k_zero=None if self.k_zero is None else self.k_zero[layer_idx, :, :, :qgroups, :],
            v_q=(
                None
                if self.v_q is None
                else self.v_q[layer_idx, :, :, : self.quantized_length, :, :]
            ),
            v_scale=(
                None
                if self.v_scale is None
                else self.v_scale[layer_idx, :, :, : self.quantized_length, :]
            ),
            v_zero=(
                None
                if self.v_zero is None
                else self.v_zero[layer_idx, :, :, : self.quantized_length, :]
            ),
            k_fp=None if self.k_fp is None else self.k_fp[layer_idx, :, :, : self.length, :],
            v_fp=None if self.v_fp is None else self.v_fp[layer_idx, :, :, : self.length, :],
            k_residual=(
                None
                if self.k_residual is None
                else self.k_residual[layer_idx, :, :, : self.residual_length, :]
            ),
            v_residual=(
                None
                if self.v_residual is None
                else self.v_residual[layer_idx, :, :, : self.residual_length, :]
            ),
            head_dim=self.head_dim,
        )

    def dequantize_layer(
        self, layer_idx: int, *, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = dtype or self.compute_dtype
        view = self.layer_view(layer_idx)
        if view.k_bits == 16:
            assert view.k_fp is not None
            key = view.k_fp.to(dtype)
        else:
            parts = []
            if view.quantized_length:
                assert view.k_q is not None and view.k_scale is not None and view.k_zero is not None
                parts.append(
                    dequantize_keys(
                        PackedKeys(
                            payload=view.k_q,
                            scale=view.k_scale,
                            zero=view.k_zero,
                            bits=view.k_bits,
                            token_group=view.key_token_group,
                            tokens=view.quantized_length,
                        ),
                        dtype=dtype,
                    )
                )
            if view.residual_length:
                assert view.k_residual is not None
                parts.append(view.k_residual.to(dtype))
            key = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)

        if view.v_bits == 16:
            assert view.v_fp is not None
            value = view.v_fp.to(dtype)
        else:
            parts_v = []
            if view.quantized_length:
                assert view.v_q is not None and view.v_scale is not None and view.v_zero is not None
                parts_v.append(
                    dequantize_values(
                        PackedValues(
                            payload=view.v_q,
                            scale=view.v_scale,
                            zero=view.v_zero,
                            bits=view.v_bits,
                            channel_group=view.value_channel_group,
                            tokens=view.quantized_length,
                        ),
                        dtype=dtype,
                    )
                )
            if view.residual_length:
                assert view.v_residual is not None
                parts_v.append(view.v_residual.to(dtype))
            value = parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=2)
        return key, value

    def gather_layer_reference(
        self,
        layer_idx: int,
        indices: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.dequantize_layer(layer_idx, dtype=dtype)
        return gather_per_kv_head(key, indices), gather_per_kv_head(value, indices)

    def logical_nbytes(self) -> dict[str, int]:
        l, b, h, d = self.num_layers, self.batch_size, self.num_kv_heads, self.head_dim
        qn = self.quantized_length
        rn = self.residual_length
        param_size = torch.tensor([], dtype=self.param_dtype).element_size()
        compute_size = torch.tensor([], dtype=self.compute_dtype).element_size()

        if self.quant.k_bits == 16:
            k_payload = l * b * h * self.length * d * compute_size
            k_meta = 0
            k_res = 0
        else:
            k_payload = l * b * h * qn * d * self.quant.k_bits // 8
            k_meta = (
                l
                * b
                * h
                * (qn // self.quant.key_token_group)
                * d
                * 2
                * param_size
            )
            k_res = l * b * h * rn * d * compute_size

        if self.quant.v_bits == 16:
            v_payload = l * b * h * self.length * d * compute_size
            v_meta = 0
            v_res = 0
        else:
            v_payload = l * b * h * qn * d * self.quant.v_bits // 8
            v_meta = (
                l
                * b
                * h
                * qn
                * (d // self.quant.value_channel_group)
                * 2
                * param_size
            )
            v_res = l * b * h * rn * d * compute_size

        return {
            "key_payload": int(k_payload),
            "key_metadata": int(k_meta),
            "key_residual": int(k_res),
            "value_payload": int(v_payload),
            "value_metadata": int(v_meta),
            "value_residual": int(v_res),
            "total": int(k_payload + k_meta + k_res + v_payload + v_meta + v_res),
        }
