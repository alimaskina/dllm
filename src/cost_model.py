"""Idealized hardware cost accounting for sparse + quantized attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepCost:
    phase: str  # selector | exec
    num_queries: int
    old_cache_len: int
    selected_k: int
    current_block_len: int
    sparse: bool
    q_bits: int
    k_bits: int
    v_bits: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int

    @property
    def old_k_len(self) -> int:
        return self.selected_k if self.sparse else self.old_cache_len

    @property
    def total_k_len(self) -> int:
        return self.old_k_len + self.current_block_len

    @property
    def qk_macs_dense_old(self) -> int:
        """MACs for QK on full old cache (all layers)."""
        return (
            self.num_layers
            * self.num_queries
            * self.old_cache_len
            * self.num_q_heads
            * self.head_dim
        )

    @property
    def qk_macs_sparse_old(self) -> int:
        if not self.sparse:
            return self.qk_macs_dense_old
        return (
            self.num_layers
            * self.num_queries
            * self.selected_k
            * self.num_q_heads
            * self.head_dim
        )

    @property
    def qk_macs_current_block(self) -> int:
        return (
            self.num_layers
            * self.num_queries
            * self.current_block_len
            * self.num_q_heads
            * self.head_dim
        )

    @property
    def pv_macs_old(self) -> int:
        macs = (
            self.num_layers
            * self.num_queries
            * self.old_k_len
            * self.num_q_heads
            * self.head_dim
        )
        return macs

    @property
    def pv_macs_current(self) -> int:
        return (
            self.num_layers
            * self.num_queries
            * self.current_block_len
            * self.num_q_heads
            * self.head_dim
        )

    @property
    def kv_bytes_read_old(self) -> int:
        """Bytes read from old KV for this step (K+V, all layers)."""
        elems = self.num_layers * self.num_kv_heads * self.old_k_len * self.head_dim
        k_bytes = elems * max(self.k_bits, 1) // 8
        v_bytes = elems * max(self.v_bits, 1) // 8
        if self.k_bits >= 16:
            k_bytes = elems * 2
        if self.v_bits >= 16:
            v_bytes = elems * 2
        return k_bytes + v_bytes

    @property
    def kv_stored_bytes_old(self) -> int:
        elems = self.num_layers * self.num_kv_heads * self.old_cache_len * self.head_dim
        k_b = elems * (2 if self.k_bits >= 16 else max(self.k_bits, 1) // 8)
        v_b = elems * (2 if self.v_bits >= 16 else max(self.v_bits, 1) // 8)
        return k_b + v_b

    @property
    def bit_weighted_qk(self) -> float:
        qk = self.qk_macs_sparse_old + self.qk_macs_current_block
        return qk * (self.q_bits / 16.0) * (self.k_bits / 16.0)

    @property
    def bit_weighted_pv(self) -> float:
        pv = self.pv_macs_old + self.pv_macs_current
        return pv * (self.v_bits / 16.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "num_queries": self.num_queries,
            "old_cache_len": self.old_cache_len,
            "selected_k": self.selected_k,
            "current_block_len": self.current_block_len,
            "sparse": self.sparse,
            "q_bits": self.q_bits,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "qk_macs_dense_old": self.qk_macs_dense_old,
            "qk_macs_sparse_old": self.qk_macs_sparse_old,
            "qk_macs_current_block": self.qk_macs_current_block,
            "pv_macs_old": self.pv_macs_old,
            "pv_macs_current": self.pv_macs_current,
            "kv_bytes_read_old": self.kv_bytes_read_old,
            "kv_stored_bytes_old": self.kv_stored_bytes_old,
            "bit_weighted_qk": self.bit_weighted_qk,
            "bit_weighted_pv": self.bit_weighted_pv,
        }


@dataclass
class RunCostSummary:
    step_costs: list[StepCost] = field(default_factory=list)
    dense_fp16_baseline: dict[str, float] = field(default_factory=dict)

    def add(self, step: StepCost) -> None:
        self.step_costs.append(step)

    def finalize_baseline(self) -> None:
        """Use first dense selector step as FP16 baseline reference."""
        dense = [s for s in self.step_costs if not s.sparse and s.q_bits >= 16 and s.k_bits >= 16]
        if not dense:
            dense = self.step_costs[:1]
        if not dense:
            return
        ref = dense[0]
        self.dense_fp16_baseline = {
            "qk_macs": ref.qk_macs_dense_old + ref.qk_macs_current_block,
            "pv_macs": ref.pv_macs_old + ref.pv_macs_current,
            "kv_bytes_read": ref.kv_bytes_read_old,
            "kv_stored": ref.kv_stored_bytes_old,
            "bit_weighted_qk": ref.bit_weighted_qk,
            "bit_weighted_pv": ref.bit_weighted_pv,
        }

    def dense_cost_at_cache_len(
        self,
        old_cache_len: int,
        *,
        num_queries: int | None = None,
        current_block_len: int | None = None,
    ) -> dict[str, float]:
        """Reference dense FP16 cost at a given old-cache length."""
        ref = next((s for s in self.step_costs if s.old_cache_len == old_cache_len), None)
        if ref is None:
            ref = self.step_costs[0] if self.step_costs else None
        if ref is None:
            return {}
        nq = num_queries or ref.num_queries
        cbl = current_block_len if current_block_len is not None else ref.current_block_len
        nl = ref.num_layers
        nqh = ref.num_q_heads
        nkv = ref.num_kv_heads
        hd = ref.head_dim
        qk = nl * nq * old_cache_len * nqh * hd + nl * nq * cbl * nqh * hd
        pv = nl * nq * old_cache_len * nqh * hd + nl * nq * cbl * nqh * hd
        elems = nl * nkv * old_cache_len * hd
        kv_bytes = elems * 2 * 2  # K+V fp16
        return {
            "old_cache_len": old_cache_len,
            "qk_macs": float(qk),
            "pv_macs": float(pv),
            "kv_bytes_read": float(kv_bytes),
            "bit_weighted_qk": float(qk),
            "bit_weighted_pv": float(pv),
        }

    def aggregate_by_cache_len(self) -> dict[str, Any]:
        """Per old-cache-length idealized speedup (exec steps only)."""
        exec_steps = [s for s in self.step_costs if s.phase == "exec" and s.old_cache_len > 0]
        if not exec_steps:
            exec_steps = [s for s in self.step_costs if s.old_cache_len > 0]
        by_len: dict[int, list[StepCost]] = {}
        for s in exec_steps:
            by_len.setdefault(s.old_cache_len, []).append(s)

        out: dict[str, Any] = {}
        for cache_len, steps in sorted(by_len.items()):
            dense = self.dense_cost_at_cache_len(cache_len)
            if not dense:
                continue

            def _avg(attr: str) -> float:
                return float(sum(getattr(s, attr) for s in steps) / len(steps))

            act_qk = _avg("qk_macs_sparse_old") + _avg("qk_macs_current_block")
            act_pv = _avg("pv_macs_old") + _avg("pv_macs_current")
            act_kv = _avg("kv_bytes_read_old")
            act_bwqk = _avg("bit_weighted_qk")
            act_bwpv = _avg("bit_weighted_pv")

            out[str(cache_len)] = {
                "old_cache_len": cache_len,
                "num_exec_steps": len(steps),
                "avg_selected_k": _avg("selected_k"),
                "effective_sparsity": _avg("selected_k") / max(cache_len, 1),
                "actual_qk_macs": act_qk,
                "actual_pv_macs": act_pv,
                "actual_kv_bytes_read": act_kv,
                "actual_bit_weighted_qk": act_bwqk,
                "actual_bit_weighted_pv": act_bwpv,
                "speedup_vs_dense_fp16": {
                    "qk_macs": dense["qk_macs"] / max(act_qk, 1),
                    "pv_macs": dense["pv_macs"] / max(act_pv, 1),
                    "kv_bytes_read": dense["kv_bytes_read"] / max(act_kv, 1),
                    "bit_weighted_qk": dense["bit_weighted_qk"] / max(act_bwqk, 1),
                    "bit_weighted_pv": dense["bit_weighted_pv"] / max(act_bwpv, 1),
                },
            }
        return out

    def aggregate(self, *, save_full_steps: bool = False) -> dict[str, Any]:
        self.finalize_baseline()
        sel = [s for s in self.step_costs if s.phase == "selector"]
        exe = [s for s in self.step_costs if s.phase == "exec"]

        def _avg(attr: str, steps: list[StepCost]) -> float:
            if not steps:
                return 0.0
            return float(sum(getattr(s, attr) for s in steps) / len(steps))

        base = self.dense_fp16_baseline
        if not base:
            ref = sel[0] if sel else (exe[0] if exe else None)
            if ref:
                base = {
                    "qk_macs": ref.qk_macs_dense_old + ref.qk_macs_current_block,
                    "pv_macs": ref.pv_macs_old + ref.pv_macs_current,
                    "kv_bytes_read": ref.kv_bytes_read_old,
                    "kv_stored": ref.kv_stored_bytes_old,
                    "bit_weighted_qk": ref.bit_weighted_qk,
                    "bit_weighted_pv": ref.bit_weighted_pv,
                }

        avg_exec_qk = _avg("qk_macs_sparse_old", exe) + _avg("qk_macs_current_block", exe)
        avg_exec_pv = _avg("pv_macs_old", exe) + _avg("pv_macs_current", exe)
        avg_exec_kv = _avg("kv_bytes_read_old", exe)
        avg_sel_qk = _avg("qk_macs_dense_old", sel) + _avg("qk_macs_current_block", sel)
        avg_sel_kv = _avg("kv_bytes_read_old", sel)

        dense_qk = base.get("qk_macs", avg_sel_qk or 1)
        dense_pv = base.get("pv_macs", 1)
        dense_kv = base.get("kv_bytes_read", avg_sel_kv or 1)

        return {
            "num_steps_recorded": len(self.step_costs),
            "selector_steps": len(sel),
            "exec_steps": len(exe),
            "avg_selector_qk_macs": avg_sel_qk,
            "avg_exec_qk_macs": avg_exec_qk,
            "avg_exec_pv_macs": avg_exec_pv,
            "avg_exec_kv_bytes_read": avg_exec_kv,
            "avg_selector_kv_bytes_read": avg_sel_kv,
            "total_bit_weighted_qk": sum(s.bit_weighted_qk for s in self.step_costs),
            "total_bit_weighted_pv": sum(s.bit_weighted_pv for s in self.step_costs),
            "ratio_vs_dense_fp16_per_step": {
                "exec_qk_macs": avg_exec_qk / max(dense_qk, 1),
                "exec_pv_macs": avg_exec_pv / max(dense_pv, 1),
                "exec_kv_bytes_read": avg_exec_kv / max(dense_kv, 1),
                "selector_qk_macs": avg_sel_qk / max(dense_qk, 1),
            },
            "idealized_speedup_per_step": {
                "exec_qk_macs": 1.0 / max(avg_exec_qk / max(dense_qk, 1), 1e-9),
                "exec_kv_bytes_read": 1.0 / max(avg_exec_kv / max(dense_kv, 1), 1e-9),
                "exec_bit_weighted_qk": 1.0
                / max(_avg("bit_weighted_qk", exe) / max(base.get("bit_weighted_qk", 1), 1), 1e-9),
            },
            "by_cache_len": self.aggregate_by_cache_len(),
            "per_step": [s.to_dict() for s in (self.step_costs if save_full_steps else self.step_costs[:20])],
            "per_step_truncated": (not save_full_steps) and len(self.step_costs) > 20,
        }
