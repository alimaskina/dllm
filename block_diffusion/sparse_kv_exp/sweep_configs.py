"""Adaptive sweep: extremes first, skip expensive configs if already near-ideal."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from config import ExperimentConfig, PrecisionConfig, SelectorConfig


@dataclass
class SweepEntry:
    cfg: ExperimentConfig
    tier: str  # baseline | extreme | medium | expensive
    topk: int
    mode_key: str
    exec_label: str
    inference_cost_rank: int  # lower = cheaper inference (run first)


def _mode_key(mode: str, uniform_n: int = 5) -> str:
    return mode if mode != "uniform" else f"uniform{uniform_n}"


def _bit_cost(k: str, v: str, q: str = "fp16") -> int:
    def _b(x: str) -> int:
        s = str(x).lower()
        if s in ("fp16", "bf16", "16"):
            return 16
        return int(s)

    return _b(k) + _b(v) + _b(q)


def _make_sparse_config(
    *,
    topk: int,
    mode: str,
    mode_kw: dict,
    exec_label: str,
    sel_k: str,
    exec_k: str,
    exec_v: str,
    exec_q: str,
    tier: str,
) -> SweepEntry:
    uniform_n = mode_kw.get("uniform_n", 5)
    mode_suffix = _mode_key(mode, uniform_n)
    name = f"sparse_k{topk}_{mode_suffix}_{exec_label}"

    sel_prec = PrecisionConfig(k_bits=sel_k, v_bits="fp16", q_bits="fp16")
    exec_prec = PrecisionConfig(k_bits=exec_k, v_bits=exec_v, q_bits=exec_q)

    baseline = "sparse_fp16"
    if exec_k != "fp16" or exec_v != "fp16":
        baseline = "sparse_quant_kv"
    if exec_q != "fp16":
        baseline = "sparse_quant_kvq"

    # Cheaper inference = lower rank (smaller topk, lower bits)
    cost_rank = topk * 100 + _bit_cost(exec_k, exec_v, exec_q)

    return SweepEntry(
        cfg=ExperimentConfig(
            name=name,
            baseline=baseline,
            sparse_old_cache=True,
            selector=SelectorConfig(mode=mode, uniform_n=uniform_n, topk=topk),
            selector_precision=sel_prec,
            exec_precision=exec_prec,
            save_full_cost_steps=True,
        ),
        tier=tier,
        topk=topk,
        mode_key=mode_suffix,
        exec_label=exec_label,
        inference_cost_rank=cost_rank,
    )


def build_adaptive_sweep() -> list[SweepEntry]:
    """Ordered sweep: baseline → extreme → medium → expensive."""
    entries: list[SweepEntry] = []

    entries.append(
        SweepEntry(
            cfg=ExperimentConfig(
                name="baseline_dense_fp16",
                baseline="original",
                sparse_old_cache=False,
                save_full_cost_steps=True,
            ),
            tier="baseline",
            topk=0,
            mode_key="—",
            exec_label="fp16",
            inference_cost_rank=0,
        )
    )

    modes = [
        ("all_mean", {}),
        ("middle", {}),
        ("uniform", {"uniform_n": 5}),
    ]

    # (label, sel_k, exec_k, exec_v, exec_q, tier)
    exec_specs = [
        ("k2v2", "fp16", "2", "2", "fp16", "extreme"),
        ("k4v2", "fp16", "4", "2", "fp16", "extreme"),
        ("k4v4_q4", "fp16", "4", "4", "4", "extreme"),
        ("k4v4", "fp16", "4", "4", "fp16", "medium"),
        ("fp16", "fp16", "fp16", "fp16", "fp16", "expensive"),
    ]

    topk_tiers = [
        (128, "extreme"),
        (256, "medium"),
        (512, "expensive"),
    ]

    for topk, topk_tier in topk_tiers:
        for (mode, mode_kw), (label, sk, ek, ev, eq, exec_tier) in product(modes, exec_specs):
            tier = exec_tier if topk_tier == "extreme" else (
                "medium" if topk_tier == "medium" and exec_tier != "expensive" else
                "expensive" if topk_tier == "expensive" or exec_tier == "expensive" else
                "medium"
            )
            # Simplify tier assignment
            if topk == 128 and label in ("k2v2", "k4v2", "k4v4_q4"):
                tier = "extreme"
            elif topk == 512 or label == "fp16":
                tier = "expensive"
            elif topk == 256 or label == "k4v4":
                tier = "medium"
            else:
                tier = "extreme"

            entries.append(
                _make_sparse_config(
                    topk=topk,
                    mode=mode,
                    mode_kw=mode_kw,
                    exec_label=label,
                    sel_k=sk,
                    exec_k=ek,
                    exec_v=ev,
                    exec_q=eq,
                    tier=tier,
                )
            )

    # Sort: baseline first, then by inference_cost_rank
    baseline = [e for e in entries if e.tier == "baseline"]
    rest = sorted([e for e in entries if e.tier != "baseline"], key=lambda e: e.inference_cost_rank)
    return baseline + rest


def build_sweep_grid(*, quick: bool = False) -> list[ExperimentConfig]:
    """Legacy API — returns configs only."""
    entries = build_adaptive_sweep()
    if quick:
        # minimal: baseline + 3 extremes
        names = {
            "baseline_dense_fp16",
            "sparse_k128_all_mean_k2v2",
            "sparse_k128_middle_k4v2",
            "sparse_k128_uniform5_k4v4_q4",
        }
        return [e.cfg for e in entries if e.cfg.name in names]
    return [e.cfg for e in entries]


# ── early-stop logic ─────────────────────────────────────────────────────

IDEAL_COVERAGE = 0.95
IDEAL_ACCURACY_DROP = 0.0  # must match baseline exactly on small n


def _avg_coverage(record: dict) -> float | None:
    masses = []
    for blk in record.get("blocks", []):
        for layer in blk.get("layers", {}).values():
            m = layer.get("attention_mass_captured")
            if m is not None:
                masses.append(m)
    return sum(masses) / len(masses) if masses else None


def config_metrics(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    coverages = [c for r in records if (c := _avg_coverage(r)) is not None]
    return {
        "accuracy": correct / n,
        "avg_coverage": sum(coverages) / len(coverages) if coverages else None,
        "num_examples": n,
    }


def is_near_ideal(metrics: dict[str, Any], baseline_accuracy: float) -> bool:
    acc = metrics.get("accuracy")
    cov = metrics.get("avg_coverage")
    if acc is None or cov is None:
        return False
    if acc < baseline_accuracy - IDEAL_ACCURACY_DROP:
        return False
    return cov >= IDEAL_COVERAGE


def should_skip_entry(
    entry: SweepEntry,
    *,
    baseline_accuracy: float,
    completed_by_config: dict[str, list[dict]],
    skip_log: list[str] | None = None,
) -> bool:
    """Skip expensive configs when a cheaper extreme already looks ideal."""
    if entry.tier in ("baseline", "extreme"):
        return False

    def _log(msg: str) -> None:
        if skip_log is not None:
            skip_log.append(msg)

    # Find best completed extreme for same (mode, exec_label)
    extremes = [
        e for e in build_adaptive_sweep()
        if e.tier == "extreme"
        and e.mode_key == entry.mode_key
        and e.exec_label == entry.exec_label
        and e.topk <= entry.topk
    ]
    for ex in extremes:
        recs = completed_by_config.get(ex.cfg.name, [])
        if len(recs) < 1:
            continue
        m = config_metrics(recs)
        if is_near_ideal(m, baseline_accuracy):
            if entry.topk > ex.topk:
                _log(
                    f"SKIP {entry.cfg.name}: topk={ex.topk} {ex.exec_label} "
                    f"{ex.mode_key} already ideal (acc={m['accuracy']:.2f}, cov={m['avg_coverage']:.3f})"
                )
                return True

    # Same topk+mode: skip expensive quant if k2v2 extreme ideal
    cheaper_quants = ["k2v2", "k4v2", "k4v4_q4"]
    expensive_quants = ["k4v4", "fp16"]
    if entry.exec_label in expensive_quants:
        for label in cheaper_quants:
            cheaper = [
                e for e in build_adaptive_sweep()
                if e.topk == entry.topk
                and e.mode_key == entry.mode_key
                and e.exec_label == label
                and e.tier == "extreme"
            ]
            for c in cheaper:
                recs = completed_by_config.get(c.cfg.name, [])
                if not recs:
                    continue
                m = config_metrics(recs)
                if is_near_ideal(m, baseline_accuracy):
                    _log(
                        f"SKIP {entry.cfg.name}: cheaper {c.cfg.name} ideal "
                        f"(acc={m['accuracy']:.2f}, cov={m['avg_coverage']:.3f})"
                    )
                    return True

    return False
