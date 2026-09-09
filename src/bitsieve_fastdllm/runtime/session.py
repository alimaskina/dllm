from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cache import LayerCacheView, PackedKVCache
from ..config import ExperimentConfig
from ..kernels.ops import (
    SelectorScratch,
    dense_packed_attention,
    gather_packed_kv,
    quantize_key_groups_into,
    quantize_values_into,
    selector_topk,
)
from ..reference import sdpa_compact
from .trace import RunTrace, RuntimeTimer, SelectionRecord


@dataclass(slots=True)
class CompactLayerState:
    key: torch.Tensor
    value: torch.Tensor
    indices: torch.Tensor | None = None
    selected_k: int = 0
    use_dense: bool = False
    ready_event: torch.cuda.Event | None = None
    pending_refs: tuple[torch.Tensor, ...] | None = None
    packed_view: LayerCacheView | None = None


class BitSieveSession:

    def __init__(
        self,
        cache: PackedKVCache,
        config: ExperimentConfig,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compute_dtype: torch.dtype,
    ) -> None:
        config.validate(head_dim=head_dim, num_layers=cache.num_layers)
        self.cache = cache
        self.config = config
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.compute_dtype = compute_dtype
        self.trace = RunTrace()
        self.timer = RuntimeTimer(self.trace, enabled=config.profile_layers)
        self.selector_scratch = SelectorScratch()

        self.block_index = -1
        self.step_index = -1
        self.masked_positions: list[int] = []
        self.query_indices: list[int] = []
        self.commit_current = False
        self._step0_finished = False
        self._states: dict[int, CompactLayerState] = {}
        self._selection_stream: torch.cuda.Stream | None = None
        if cache.device.type == "cuda" and config.async_selector and config.semantic == "A":


            self._selection_stream = torch.cuda.Stream(device=cache.device)

    @property
    def old_cache_len(self) -> int:
        return self.cache.get_seq_length()

    def begin_block(self, masked_positions: list[int]) -> None:
        self.block_index += 1
        self.step_index = -1
        self.masked_positions = sorted(set(masked_positions))
        self.query_indices = self.config.selector.query_indices(
            self.masked_positions, self.config.generation.block_size
        )
        self.commit_current = False
        self._step0_finished = False


        for state in self._states.values():
            if state.ready_event is not None:
                raise RuntimeError("begin_block called with unfinished selector work")
            state.indices = None
            state.selected_k = 0
            state.use_dense = False
            state.ready_event = None
            state.pending_refs = None
            state.packed_view = None
        self.trace.add_counter("blocks", 1)

    def prepare_forward(
        self,
        step_index: int,
        *,
        masked_positions: list[int] | None = None,
        commit_current: bool = False,
    ) -> None:
        self.step_index = int(step_index)
        if masked_positions is not None:
            self.masked_positions = sorted(set(int(x) for x in masked_positions))
        self.commit_current = bool(commit_current)

    def _max_selected(self) -> int:
        selector = self.config.selector
        if selector.topk_percent is None:
            return min(selector.topk, self.config.max_cache_tokens)
        return min(
            self.config.max_cache_tokens,
            max(1, int(self.config.max_cache_tokens * selector.topk_percent / 100.0 + 0.999)),
        )

    def _ensure_state(self, layer_idx: int, selected_k: int) -> CompactLayerState:
        state = self._states.get(layer_idx)
        capacity = max(1, selected_k) + self.config.generation.block_size
        if state is not None and state.key.shape[2] >= capacity:
            return state
        shape = (
            self.cache.batch_size,
            self.num_kv_heads,
            capacity,
            self.head_dim,
        )
        state = CompactLayerState(
            key=torch.empty(shape, device=self.cache.device, dtype=self.compute_dtype),
            value=torch.empty(shape, device=self.cache.device, dtype=self.compute_dtype),
        )
        self._states[layer_idx] = state
        return state

    def _dense_required(self, layer_idx: int) -> bool:
        if self.config.semantic == "dense":
            return True
        if layer_idx < self.config.selector.dense_prefix_layers:
            return True
        k = self.config.selector.effective_topk(self.old_cache_len)
        return self.old_cache_len == 0 or k >= self.old_cache_len or not self.query_indices

    def _requantize_compact(self, state: CompactLayerState) -> None:
        k = state.selected_k
        if k <= 0:
            raise RuntimeError("cannot requantize an empty selection")
        bits_k = self.config.quant.k_bits
        bits_v = self.config.quant.v_bits
        group_k = self.config.quant.key_token_group
        group_v = self.config.quant.value_channel_group
        qn = (k // group_k) * group_k if bits_k < 16 or bits_v < 16 else k
        rn = k - qn
        b, h, _, d = state.key.shape
        device = state.key.device
        param_dtype = self.cache.param_dtype

        kq = ks = kz = vq = vs = vz = kfp = vfp = kr = vr = None
        if bits_k == 16:
            kfp = state.key[:, :, :k, :].clone()
        else:
            if qn:
                vpb = 8 // bits_k
                kq = torch.empty((b, h, qn // group_k, d, group_k // vpb), device=device, dtype=torch.uint8)
                ks = torch.empty((b, h, qn // group_k, d), device=device, dtype=param_dtype)
                kz = torch.empty_like(ks)
                quantize_key_groups_into(
                    state.key[:, :, :qn, :].contiguous(),
                    kq, ks, kz, bits=bits_k, token_group=group_k, backend=self.config.backend
                )
            if rn:
                kr = state.key[:, :, qn:k, :].clone()

        if bits_v == 16:
            vfp = state.value[:, :, :k, :].clone()
        else:
            if qn:
                vpb_v = 8 // bits_v
                vg = d // group_v
                vq = torch.empty((b, h, qn, vg, group_v // vpb_v), device=device, dtype=torch.uint8)
                vs = torch.empty((b, h, qn, vg), device=device, dtype=param_dtype)
                vz = torch.empty_like(vs)
                quantize_values_into(
                    state.value[:, :, :qn, :].contiguous(),
                    vq, vs, vz, bits=bits_v, channel_group=group_v, backend=self.config.backend
                )
            if rn:
                vr = state.value[:, :, qn:k, :].clone()

        state.packed_view = LayerCacheView(
            k_bits=bits_k,
            v_bits=bits_v,
            length=k,
            quantized_length=qn,
            residual_length=rn,
            key_token_group=group_k,
            value_channel_group=group_v,
            k_q=kq, k_scale=ks, k_zero=kz,
            v_q=vq, v_scale=vs, v_zero=vz,
            k_fp=kfp, v_fp=vfp,
            k_residual=kr, v_residual=vr,
            head_dim=d,
        )

    def _select_and_gather(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        state: CompactLayerState,
    ) -> None:
        k = self.config.selector.effective_topk(self.old_cache_len)
        view = self.cache.layer_view(layer_idx)
        with self.timer.region(
            "selector",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
            stream_name="selector" if self._selection_stream is not None else "main",
            stream=self._selection_stream,
        ):
            selected = selector_topk(
                query,
                view,
                query_indices=self.query_indices,
                topk=k,
                current_key=current_key,
                domain=self.config.selector.domain,
                score_kind=self.config.selector.score,
                sort_indices=self.config.selector.sort_indices,
                scaling=self.head_dim**-0.5,
                backend=self.config.backend,
                return_importance=self.config.collect_diagnostics,
                scratch=self.selector_scratch,
                kernel_variant=self.config.selector_kernel_variant,
                logits_dtype=(
                    torch.float32
                    if self.config.selector_logits_dtype == "float32"
                    else torch.float16
                ),
            )
        with self.timer.region(
            "gather_dequant",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
            stream_name="selector" if self._selection_stream is not None else "main",
            stream=self._selection_stream,
        ):
            gather_packed_kv(
                view,
                selected.indices,
                dtype=self.compute_dtype,
                backend=self.config.backend,
                out_key=state.key[:, :, :k, :],
                out_value=state.value[:, :, :k, :],
            )
        state.indices = selected.indices
        state.selected_k = k
        if self.config.compact_format == "requantized":
            with self.timer.region(
                "requantize_compact",
                tensor=query,
                layer=layer_idx,
                block=self.block_index,
                step=self.step_index,
                stream_name="selector" if self._selection_stream is not None else "main",
                stream=self._selection_stream,
            ):
                self._requantize_compact(state)
        if self.config.collect_diagnostics:

            checksum = int(selected.indices.to(torch.int64).sum().item())
            mean_score = (
                float(selected.values.float().mean().item()) if selected.values.numel() else None
            )
        else:
            checksum = 0
            mean_score = None
        self.trace.selections.append(
            SelectionRecord(
                block=self.block_index,
                layer=layer_idx,
                old_cache_len=self.old_cache_len,
                selected_k=k,
                selector_queries=list(self.query_indices),
                semantic=self.config.semantic,
                index_checksum=checksum,
                mean_score=mean_score,
            )
        )

    def _schedule_a_selection(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
    ) -> None:
        k = self.config.selector.effective_topk(self.old_cache_len)
        state = self._ensure_state(layer_idx, k)
        if state.indices is not None or state.ready_event is not None:
            return
        if self._selection_stream is None:
            self._select_and_gather(layer_idx, query, current_key, state)
            return

        main_stream = torch.cuda.current_stream(query.device)
        q_ready = torch.cuda.Event()
        q_ready.record(main_stream)
        with torch.cuda.stream(self._selection_stream):
            self._selection_stream.wait_event(q_ready)
            self._select_and_gather(layer_idx, query, current_key, state)
            done = torch.cuda.Event()
            done.record(self._selection_stream)
        state.ready_event = done

        state.pending_refs = (query, current_key)

    def finish_step0(self) -> None:
        if self._step0_finished:
            return
        if self._selection_stream is not None:
            main = torch.cuda.current_stream(self.cache.device)
            for state in self._states.values():
                if state.ready_event is not None:
                    main.wait_event(state.ready_event)
                    state.ready_event = None
                    state.pending_refs = None
        self._step0_finished = True

    def _compact_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        state = self._states.get(layer_idx)
        if state is None or state.selected_k <= 0:
            raise RuntimeError(
                f"layer {layer_idx} has no selected cache; finish_step0 was not called or selection failed"
            )
        k = state.selected_k
        ncur = current_key.shape[2]
        if self.config.compact_format == "requantized":
            if state.packed_view is None:
                raise RuntimeError("requantized compact view is missing")
            with self.timer.region(
                "compact_requantized_attention",
                tensor=query,
                layer=layer_idx,
                block=self.block_index,
                step=self.step_index,
            ):
                return dense_packed_attention(
                    query,
                    state.packed_view,
                    current_key,
                    current_value,
                    scaling=self.head_dim**-0.5,
                    backend=self.config.backend,
                    kernel_variant=self.config.dense_kernel_variant,
                )
        if k + ncur > state.key.shape[2]:
            raise RuntimeError("compact cache buffer capacity exceeded")
        state.key[:, :, k : k + ncur, :].copy_(current_key.to(self.compute_dtype))
        state.value[:, :, k : k + ncur, :].copy_(current_value.to(self.compute_dtype))
        query_compute = query.to(self.compute_dtype)
        with self.timer.region(
            "compact_attention",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            output = sdpa_compact(
                query_compute,
                state.key[:, :, : k + ncur, :],
                state.value[:, :, : k + ncur, :],
            )
        return output.to(query.dtype)

    def _dense_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.old_cache_len == 0:
            return sdpa_compact(query, current_key, current_value)
        with self.timer.region(
            "dense_packed_attention",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            return dense_packed_attention(
                query,
                self.cache.layer_view(layer_idx),
                current_key,
                current_value,
                scaling=self.head_dim**-0.5,
                backend=self.config.backend,
                kernel_variant=self.config.dense_kernel_variant,
            )

    def attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        dense = self._dense_required(layer_idx)
        if dense:
            state = self._states.get(layer_idx)
            if state is not None:
                state.use_dense = True
            return self._dense_attention(layer_idx, query, current_key, current_value)

        if self.step_index == 0:
            if self.config.semantic == "A":
                self._schedule_a_selection(layer_idx, query, current_key)
                return self._dense_attention(layer_idx, query, current_key, current_value)
            if self.config.semantic == "B":
                k = self.config.selector.effective_topk(self.old_cache_len)
                state = self._ensure_state(layer_idx, k)
                if state.selected_k == 0:
                    self._select_and_gather(layer_idx, query, current_key, state)
                return self._compact_attention(
                    layer_idx, query, current_key, current_value
                )
            raise AssertionError(f"unexpected semantic {self.config.semantic}")


        return self._compact_attention(layer_idx, query, current_key, current_value)

    def stage_if_committing(
        self,
        layer_idx: int,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> None:
        if not self.commit_current:
            return
        if not self.cache.append_in_progress:
            raise RuntimeError("commit forward ran without begin_append")
        expected = self.config.generation.block_size
        if current_key.shape[2] != expected:
            raise RuntimeError(
                f"commit forward must materialize a full block ({expected}), got {current_key.shape[2]}"
            )
        self.cache.stage_layer(
            layer_idx,
            current_key.detach(),
            current_value.detach(),
        )

    def end_block(self) -> None:
        self.finish_step0()
        self.commit_current = False

    def finalize_trace(self) -> RunTrace:
        self.timer.finalize()
        return self.trace
