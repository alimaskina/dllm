from __future__ import annotations

from typing import Any

from dataclasses import replace, dataclass

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
from ..eviction import EvictionState, live_cache_nbytes
from ..reference import sdpa_compact, simulate_key_quantization
from .trace import CoverageArmRecord, CoverageRecord, RunTrace, RuntimeTimer, SelectionRecord


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
        # Shadow fp16 keys, only when scoring coverage. Keys alone are enough:
        # attention weights depend on Q and K, not V. Seeded from the prefill
        # cache and extended at every block commit, so it mirrors exactly what
        # the packed cache holds.
        self._fp16_key_shadow: dict[int, torch.Tensor] = {}
        # Eviction: what the cache KEEPS, as opposed to what a block reads.
        self._eviction: EvictionState | None = None
        self._next_position = 0     # cache positions admitted so far
        self._tokens_seen = 0       # what the cache would hold with no eviction
        self._live_scratch: torch.Tensor | None = None
        self._live_scratch_v: torch.Tensor | None = None
        if config.eviction.enabled:
            self._eviction = EvictionState(
                num_layers=cache.num_layers,
                batch_size=cache.batch_size,
                num_kv_heads=self.num_kv_heads,
                capacity=config.eviction.capacity_floor,
                device=cache.device,
                max_position=config.max_cache_tokens,
            )

        self._selection_stream: torch.cuda.Stream | None = None
        if (
            cache.device.type == "cuda"
            and config.async_selector
            and config.semantic == "A"
            and self._eviction is None
            and not config.selector.value_aware
        ):
            # Step 0 under eviction attends over a single shared live buffer, so
            # overlapping layers on a second stream would race on it. Correctness
            # first; the overlap is an optimization, not a result.
            self._selection_stream = torch.cuda.Stream(device=cache.device)

    @property
    def old_cache_len(self) -> int:
        return self.cache.get_seq_length()

    @property
    def live_cache_len(self) -> int:
        """Entries that still exist. Equals the cache length without eviction."""
        if self._eviction is None:
            return self.old_cache_len
        return self._eviction.live

    def _admit_new_entries(self) -> None:
        """Give newly committed cache positions a slot in the live set."""
        if self._eviction is None:
            return
        length = self.cache.get_seq_length()
        if length > self._next_position:
            self._eviction.append(
                torch.arange(self._next_position, length, device=self.cache.device)
            )
            self._tokens_seen += length - self._next_position
            self._next_position = length

    def _live_positions(self, layer_idx: int) -> torch.Tensor:
        """Physical cache positions still live for one layer. [B, Hkv, live]."""
        assert self._eviction is not None
        return self._eviction.pos[layer_idx, ..., : self._eviction.live].to(torch.int64)

    def _live_mask(self, layer_idx: int) -> torch.Tensor:
        """[B, Hkv, cache_len] -- False on entries this layer has evicted."""
        assert self._eviction is not None
        n = self.cache.get_seq_length()
        mask = torch.zeros(
            (self.cache.batch_size, self.num_kv_heads, n),
            dtype=torch.bool,
            device=self.cache.device,
        )
        mask.scatter_(-1, self._live_positions(layer_idx), True)
        return mask

    def begin_block(self, masked_positions: list[int]) -> None:
        self._admit_new_entries()
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

        # Whether the selector can bite at all this block. With a fixed top-k
        # budget it cannot until the prefix outgrows k: selecting 512 of 400
        # entries IS dense attention, so such a block must not be reported as
        # sparse. Counted here so a run can never claim a sparse path it never
        # took.
        if self.config.semantic == "dense":
            self.trace.add_counter("blocks_dense_engine", 1)
        else:
            budget = self.config.selector.effective_topk(self.old_cache_len)
            engages = (
                self.old_cache_len > 0
                and bool(self.query_indices)
                and budget < self.old_cache_len
            )
            self.trace.add_counter(
                "blocks_sparse" if engages else "blocks_dense_bypass", 1
            )

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
        if self._eviction is not None:
            # With eviction the cache IS the live set, so even a budget that
            # covers it must be served from a gather -- the physical dense path
            # would read entries that no longer exist.
            return self.live_cache_len == 0 or not self.query_indices
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

    # ---- honest coverage scoring (diagnostic path only) -------------------

    def seed_fp16_key_shadow(self, layer_idx: int, key: torch.Tensor) -> None:
        """Record the prefill's fp16 keys as the coverage reference."""
        if not self.config.coverage_diagnostics:
            return
        self._fp16_key_shadow[layer_idx] = key.detach().clone()

    def _extend_fp16_key_shadow(self, layer_idx: int, key: torch.Tensor) -> None:
        if not self.config.coverage_diagnostics:
            return
        prev = self._fp16_key_shadow.get(layer_idx)
        cur = key.detach()
        self._fp16_key_shadow[layer_idx] = (
            cur.clone() if prev is None else torch.cat([prev, cur], dim=2)
        )

    @torch.no_grad()
    def _reference_importance(self, layer_idx: int, query: torch.Tensor) -> torch.Tensor | None:
        """Per-(batch, kv-head) attention mass over the prefix under exact fp16
        keys, aggregated over EVERY masked query in the block.

        This is deliberately independent of `selector.mode` and of the cache's
        bit width: it is the target the selection is trying to hit, so it must
        not inherit the candidate's handicaps. Returns [B, Hkv, N].
        """
        key = self._shadow_prefix(layer_idx)
        if key is None:
            return None
        return self._importance_from_keys(key, query)

    def _shadow_prefix(self, layer_idx: int) -> torch.Tensor | None:
        shadow = self._fp16_key_shadow.get(layer_idx)
        if shadow is None:
            return None
        n = self.old_cache_len
        if n <= 0 or shadow.shape[2] < n:
            return None
        return shadow[:, :, :n, :]

    def _importance_from_keys(
        self, key: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor | None:
        """The scoring rule, factored out so a sweep arm is scored identically.

        An arm differs from the reference in exactly one thing - the precision of
        the keys it ranks with. Sharing this function is what makes that true;
        a second copy of the softmax-mass rule would let the two drift apart and
        the comparison would quietly stop meaning what it claims.
        """
        n = key.shape[2]
        rows = self.masked_positions or list(range(query.shape[2]))
        if not rows:
            return None

        b, hq, _, d = query.shape
        hkv = int(key.shape[1])
        if hq % hkv:
            return None
        g = hq // hkv
        idx = torch.as_tensor(rows, device=query.device, dtype=torch.long)
        q = query.reshape(b, hkv, g, query.shape[2], d).index_select(3, idx)
        q = q.reshape(b, hkv, g * idx.numel(), d).float()
        k32 = key.float()
        scale = float(self.head_dim**-0.5)

        acc = torch.zeros(b, hkv, n, device=query.device, dtype=torch.float32)
        chunk = max(1, int(self.config.coverage_query_chunk))
        total = q.shape[2]
        for start in range(0, total, chunk):
            qc = q[:, :, start : start + chunk, :]
            logits = torch.matmul(qc, k32.transpose(-1, -2)) * scale
            acc += torch.softmax(logits, dim=-1).sum(dim=2)
        return acc / float(total)

    @torch.no_grad()
    def _record_coverage(
        self,
        layer_idx: int,
        query: torch.Tensor,
        indices: torch.Tensor,
        selected_k: int,
    ) -> None:
        ref = self._reference_importance(layer_idx, query)
        if ref is None:
            return
        k = min(int(selected_k), ref.shape[-1])
        if k <= 0:
            return
        ref_vals, ref_idx = torch.topk(ref, k, dim=-1)
        sel = indices.to(torch.long)
        # `ref` is a mean of softmax rows, so it sums to 1 over the prefix and
        # `got`/`best` are already shares of the FULL attention mass. Recording
        # both absolutes separates the loss the budget forces on any selector
        # from the loss this particular selector adds: a selector can be optimal
        # (mass 1.0) while the budget still costs most of the mass.
        best_abs = ref_vals.sum(dim=-1)
        got_abs = ref.gather(-1, sel).sum(dim=-1)
        best = best_abs.clamp_min(1e-12)
        got = got_abs
        mass = (got / best).flatten().tolist()
        # Index overlap against the same reference top-k.
        mark = torch.zeros_like(ref, dtype=torch.bool)
        mark.scatter_(-1, ref_idx, True)
        overlap = (mark.gather(-1, sel).sum(dim=-1).float() / float(k)).flatten().tolist()
        self._record_coverage_arms(layer_idx, query, ref)
        self.trace.coverage.append(
            CoverageRecord(
                block=self.block_index,
                layer=layer_idx,
                old_cache_len=self.old_cache_len,
                selected_k=k,
                selector_queries=len(self.query_indices),
                mass=[round(float(x), 5) for x in mass],
                overlap=[round(float(x), 5) for x in overlap],
                mass_abs=[round(float(x), 5) for x in got_abs.flatten().tolist()],
                ceiling_abs=[round(float(x), 5) for x in best_abs.flatten().tolist()],
            )
        )

    @torch.no_grad()
    def _record_coverage_arms(
        self, layer_idx: int, query: torch.Tensor, ref: torch.Tensor
    ) -> None:
        """Score each configured sweep arm against the same fp16 reference.

        Runs off the fp16 key shadow rather than the live cache, so one
        generation yields every arm on identical queries. A per-arm generation
        would diverge after the first block and the coverages would no longer be
        measurements of the same thing.

        Deliberately the slow path: simulate_key_quantization has no packed
        kernel behind it, which is what lets an arm use a bit width the kernels
        cannot address (3). It costs one scoring matmul per arm per layer-block
        and changes nothing about what the model generates.
        """
        arms = self.config.coverage_arms
        if not arms:
            return
        key = self._shadow_prefix(layer_idx)
        if key is None:
            return
        n = int(ref.shape[-1])
        group = int(self.config.quant.key_token_group)
        for arm in arms:
            bits = int(arm.get("bits", 16))
            if arm.get("topk") is not None:
                k = int(arm["topk"])
            else:
                k = int(n * float(arm["topk_percent"]) / 100.0)
            k = max(1, min(k, n))
            scored = self._importance_from_keys(
                simulate_key_quantization(
                    key, bits=bits, token_group=group, allow_ragged=True
                ),
                query,
            )
            if scored is None:
                continue
            sel = torch.topk(scored, k, dim=-1).indices
            ref_vals, ref_idx = torch.topk(ref, k, dim=-1)
            best_abs = ref_vals.sum(dim=-1)
            got_abs = ref.gather(-1, sel).sum(dim=-1)
            mark = torch.zeros_like(ref, dtype=torch.bool)
            mark.scatter_(-1, ref_idx, True)
            overlap = mark.gather(-1, sel).sum(dim=-1).float() / float(k)
            self.trace.coverage_arms.append(
                CoverageArmRecord(
                    arm=str(arm["name"]),
                    bits=bits,
                    block=self.block_index,
                    layer=layer_idx,
                    old_cache_len=self.old_cache_len,
                    selected_k=k,
                    mass=[round(float(x), 5) for x in (got_abs / best_abs.clamp_min(1e-12)).flatten().tolist()],
                    mass_abs=[round(float(x), 5) for x in got_abs.flatten().tolist()],
                    ceiling_abs=[round(float(x), 5) for x in best_abs.flatten().tolist()],
                    overlap=[round(float(x), 5) for x in overlap.flatten().tolist()],
                )
            )

    def compact_nbytes(self) -> int:
        """Bytes held by the gathered per-layer compact caches.

        These live outside PackedKVCache but are resident for the whole block,
        so a memory claim that counts only the packed cache understates the
        footprint - especially at short prefixes, where the compact buffers can
        outweigh what they were gathered from.
        """
        total = 0
        for state in self._states.values():
            total += state.key.numel() * state.key.element_size()
            total += state.value.numel() * state.value.element_size()
            view = state.packed_view
            if view is None:
                continue
            for t in (
                view.k_q, view.k_scale, view.k_zero,
                view.v_q, view.v_scale, view.v_zero,
                view.k_fp, view.v_fp, view.k_residual, view.v_residual,
            ):
                if t is not None:
                    total += t.numel() * t.element_size()
        return int(total)

    def _effective_budget(self) -> int:
        """Entries a block may read. Never more than the live set."""
        live = self.live_cache_len
        return min(self.config.selector.effective_topk(live), live)

    def _rescore_by_value(self, layer_idx, selected, k, live_mask):
        """Re-rank the candidates by importance * ||v - v_head_mean||.

        The plain selector keeps whatever the attention mass alone points at.
        This weighs each candidate by how far its value sits from the head's
        mean value, so an entry that is attended to but carries the same thing
        the head already averages in loses its slot to one that does not.

        Also counts how much this actually changes the kept set, because a
        rescoring that reorders nothing cannot change the output either.
        """
        imp = selected.importance
        if imp is None:
            raise ValueError("value_aware selection needs the selector's importance")
        n = imp.shape[-1]
        _, value = self.cache.dequantize_layer(layer_idx)
        v = value[:, :, :n, :].to(torch.float32)
        spread = (v - v.mean(dim=2, keepdim=True)).norm(dim=-1)
        score = imp.to(torch.float32) * spread
        if live_mask is not None:
            score = score.masked_fill(~live_mask[..., :n], float("-inf"))
        idx = torch.topk(score, k, dim=-1).indices
        if self.config.selector.sort_indices:
            idx = idx.sort(dim=-1).values
        old = selected.indices
        if old is not None and old.shape == idx.shape:
            same = (
                (idx.unsqueeze(-1) == old.unsqueeze(-2)).any(dim=-1).to(torch.float32).mean()
            )
            self.trace.add_counter("value_rescore_calls", 1)
            self.trace.add_counter("value_rescore_kept_ppm", int(round(float(same) * 1e6)))
        return replace(selected, indices=idx.to(old.dtype) if old is not None else idx)

    def _select_and_gather(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        state: CompactLayerState,
    ) -> None:
        k = self._effective_budget()
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
            live_mask = None if self._eviction is None else self._live_mask(layer_idx)
            selected = selector_topk(
                query,
                view,
                live_mask=live_mask,
                query_indices=self.query_indices,
                topk=k,
                current_key=current_key,
                domain=self.config.selector.domain,
                score_kind=self.config.selector.score,
                sort_indices=self.config.selector.sort_indices,
                scaling=self.head_dim**-0.5,
                backend=self.config.backend,
                return_importance=(
                    self.config.collect_diagnostics
                    or self._eviction is not None
                    or self.config.selector.value_aware
                ),
                scratch=self.selector_scratch,
                kernel_variant=self.config.selector_kernel_variant,
                logits_dtype=(
                    torch.float32
                    if self.config.selector_logits_dtype == "float32"
                    else torch.float16
                ),
            )
        if self.config.selector.value_aware:
            selected = self._rescore_by_value(layer_idx, selected, k, live_mask)

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
        if self._eviction is not None and selected.importance is not None:
            # alpha for the live set, in live-slot order. The selector scored
            # the physical cache, so read it back at the positions this layer
            # still holds.
            alpha = selected.importance.gather(-1, self._live_positions(layer_idx))
            self._eviction.observe(layer_idx, alpha, self.config.eviction.decay)

        state.indices = selected.indices
        state.selected_k = k
        if self.config.coverage_diagnostics:
            self._record_coverage(layer_idx, query, selected.indices, k)
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
        k = self._effective_budget()
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
        if self._eviction is not None and self.live_cache_len < self.old_cache_len:
            # Semantic A attends densely at step 0 -- over the *cache*, and
            # under eviction the cache is the live set. Reading the packed
            # buffer wholesale would attend to evicted entries and quietly
            # undo the policy.
            return self._live_dense_attention(
                layer_idx, query, current_key, current_value
            )
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

    def _live_dense_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        """Dense attention over the live set: gather it, then attend normally."""
        assert self._eviction is not None
        live = self._eviction.live
        ncur = current_key.shape[2]
        need = live + ncur
        shape = (self.cache.batch_size, self.num_kv_heads, need, self.head_dim)
        if self._live_scratch is None or self._live_scratch.shape[2] < need:
            self._live_scratch = torch.empty(
                shape, device=self.cache.device, dtype=self.compute_dtype
            )
            self._live_scratch_v = torch.empty_like(self._live_scratch)
        key_buf = self._live_scratch[:, :, :need, :]
        value_buf = self._live_scratch_v[:, :, :need, :]

        with self.timer.region(
            "live_gather_dequant",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            gather_packed_kv(
                self.cache.layer_view(layer_idx),
                self._live_positions(layer_idx).to(torch.int32),
                dtype=self.compute_dtype,
                backend=self.config.backend,
                out_key=key_buf[:, :, :live, :],
                out_value=value_buf[:, :, :live, :],
            )
        key_buf[:, :, live:need, :].copy_(current_key.to(self.compute_dtype))
        value_buf[:, :, live:need, :].copy_(current_value.to(self.compute_dtype))
        with self.timer.region(
            "live_dense_attention",
            tensor=query,
            layer=layer_idx,
            block=self.block_index,
            step=self.step_index,
        ):
            return sdpa_compact(query.to(self.compute_dtype), key_buf, value_buf)

    def attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor:
        dense = self._dense_required(layer_idx)
        self.trace.add_counter(
            "layer_steps_dense" if dense else "layer_steps_sparse", 1
        )
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
        self._extend_fp16_key_shadow(layer_idx, current_key)

    def _value_spread(self) -> torch.Tensor:
        """||v_i - vbar_head|| for every live entry. [L, B, Hkv, live].

        Computed from the cache as it is actually stored -- dequantized, so at
        4 bits this is the spread the model will really see, not the bf16 one.
        vbar is the mean over the live set only: an evicted entry must not keep
        influencing the centre it is no longer part of.
        """
        from ..reference import gather_per_kv_head

        assert self._eviction is not None
        st = self._eviction
        out = torch.empty(
            (st.num_layers, self.cache.batch_size, self.num_kv_heads, st.live),
            dtype=torch.float32,
            device=self.cache.device,
        )
        for layer_idx in range(st.num_layers):
            _, value = self.cache.dequantize_layer(layer_idx)
            live = gather_per_kv_head(
                value.to(torch.float32), self._live_positions(layer_idx)
            )
            out[layer_idx] = (live - live.mean(dim=2, keepdim=True)).norm(dim=-1)
        return out

    def end_block(self) -> None:
        self.finish_step0()
        self.commit_current = False
        if self._eviction is not None:
            self._admit_new_entries()
            cfg = self.config.eviction
            before = self._eviction.live
            if self._eviction.maybe_evict(
                cfg,
                block_index=self.block_index,
                tokens_seen=self._tokens_seen,
                value_spread_fn=(
                    self._value_spread if cfg.policy == "ema_recent_value" else None
                ),
            ) is not None:
                self.trace.add_counter("evictions", 1)
                self.trace.add_counter("entries_evicted", before - self._eviction.live)

    def eviction_metrics(self) -> dict[str, Any]:
        """What the policy kept, and what that costs. {} when eviction is off."""
        if self._eviction is None:
            return {}
        st = self._eviction
        cache_bytes = live_cache_nbytes(
            st.pos,
            live=st.live,
            head_dim=self.head_dim,
            k_bits=self.config.quant.k_bits,
            v_bits=self.config.quant.v_bits,
            key_token_group=self.config.quant.key_token_group,
            value_channel_group=self.config.quant.value_channel_group,
            param_bytes=torch.tensor([], dtype=self.cache.param_dtype).element_size(),
            compute_bytes=torch.tensor([], dtype=self.compute_dtype).element_size(),
        )
        state_bytes = st.state_nbytes()
        # What the same run would hold with no eviction: every token it ever saw.
        full = live_cache_nbytes(
            torch.arange(self._tokens_seen, device=st.pos.device, dtype=st.pos.dtype)
            .view(1, 1, 1, -1)
            .expand(st.num_layers, st.batch_size, st.num_kv_heads, self._tokens_seen)
            .contiguous(),
            live=self._tokens_seen,
            head_dim=self.head_dim,
            k_bits=self.config.quant.k_bits,
            v_bits=self.config.quant.v_bits,
            key_token_group=self.config.quant.key_token_group,
            value_channel_group=self.config.quant.value_channel_group,
            param_bytes=torch.tensor([], dtype=self.cache.param_dtype).element_size(),
            compute_bytes=torch.tensor([], dtype=self.compute_dtype).element_size(),
        ) if self._tokens_seen else 0
        total = cache_bytes + state_bytes
        return {
            "eviction_policy": self.config.eviction.policy,
            "eviction_live_entries": st.live,
            "eviction_tokens_seen": self._tokens_seen,
            "eviction_capacity": self.config.eviction.capacity(self._tokens_seen),
            "eviction_cache_bytes": cache_bytes,
            "eviction_state_bytes": state_bytes,
            "eviction_total_bytes": total,
            "eviction_unevicted_bytes": full,
            # The number the study reports: what the bounded cache costs against
            # the same run keeping everything, policy overhead included.
            "eviction_bytes_vs_unevicted": (total / full) if full else None,
            "eviction_state_overhead": (state_bytes / cache_bytes) if cache_bytes else None,
        }

    def finalize_trace(self) -> RunTrace:
        self.timer.finalize()
        return self.trace
