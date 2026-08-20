"""Handoff sweep grid: cache-% budgets × selector × quant variants."""

from __future__ import annotations

from config import ExperimentConfig
from longbench_sweep_configs import (
    TOPK_PCTS,
    _sparse,
    pct_to_label,
)

# baseline + middle + uniform5 + extreme (k2v2, k4v4) per pct tier
HANDOFF_TOPK_PCTS = TOPK_PCTS  # 2.5, 5, 10, 20
HANDOFF_NUM_EXAMPLES = 250
HANDOFF_EXTREME_QUANT = (("2", "2"), ("4", "4"))
HANDOFF_FAMILIES = [
    "baseline",
    "middle",
    "uniform5",
    "extreme_k2v2",
    "extreme_k4v4",
]

_EXTREME_FAMILY = {
    ("2", "2"): "extreme_k2v2",
    ("4", "4"): "extreme_k4v4",
}


def handoff_sweep_grid(
    topk_pcts: tuple[float, ...] | list[float] = HANDOFF_TOPK_PCTS,
) -> dict[str, ExperimentConfig]:
    from longbench_sweep_configs import _COMMON  # noqa: WPS433

    out: dict[str, ExperimentConfig] = {
        "baseline": ExperimentConfig(
            name="baseline_dense_fp16",
            baseline="original",
            sparse_old_cache=False,
            **_COMMON,
        ),
    }
    for pct in topk_pcts:
        pl = pct_to_label(pct)
        out[f"middle_{pl}"] = _sparse(
            name=f"middle_{pl}_fp16",
            mode="middle",
            topk_pct=pct,
            exec_k="fp16",
            exec_v="fp16",
        )
        out[f"uniform5_{pl}"] = _sparse(
            name=f"uniform5_{pl}_fp16",
            mode="uniform",
            uniform_n=5,
            topk_pct=pct,
            exec_k="fp16",
            exec_v="fp16",
        )
        for ek, ev in HANDOFF_EXTREME_QUANT:
            key = f"extreme_{pl}_k{ek}v{ev}"
            out[key] = _sparse(
                name=f"extreme_{pl}_all_mean_k{ek}v{ev}",
                mode="all_mean",
                topk_pct=pct,
                exec_k=ek,
                exec_v=ev,
            )
    return out


def handoff_config_list(
    topk_pcts: tuple[float, ...] | list[float] = HANDOFF_TOPK_PCTS,
) -> list[ExperimentConfig]:
    grid = handoff_sweep_grid(topk_pcts)
    order = ["baseline"]
    for pct in topk_pcts:
        pl = pct_to_label(pct)
        order.append(f"middle_{pl}")
        order.append(f"uniform5_{pl}")
        for ek, ev in HANDOFF_EXTREME_QUANT:
            order.append(f"extreme_{pl}_k{ek}v{ev}")
    return [grid[k] for k in order if k in grid]


def select_handoff_configs(
    *,
    topk_pcts: tuple[float, ...] | list[float] = HANDOFF_TOPK_PCTS,
    families: list[str] | None = None,
) -> list[ExperimentConfig]:
    fam = families or HANDOFF_FAMILIES
    grid = handoff_sweep_grid(topk_pcts)
    selected: list[ExperimentConfig] = []

    def maybe_add(cfg: ExperimentConfig) -> None:
        if cfg.name not in {c.name for c in selected}:
            selected.append(cfg)

    if "baseline" in fam:
        maybe_add(grid["baseline"])

    for pct in topk_pcts:
        pl = pct_to_label(pct)
        if "middle" in fam and f"middle_{pl}" in grid:
            maybe_add(grid[f"middle_{pl}"])
        if "uniform5" in fam and f"uniform5_{pl}" in grid:
            maybe_add(grid[f"uniform5_{pl}"])
        for ek, ev in HANDOFF_EXTREME_QUANT:
            family = _EXTREME_FAMILY[(ek, ev)]
            key = f"extreme_{pl}_k{ek}v{ev}"
            if family in fam and key in grid:
                maybe_add(grid[key])
    return selected


def handoff_run_count(
    *,
    num_examples: int,
    n_longbench_tasks: int = 4,
    n_benchmark_tasks: int = 3,
    topk_pcts: tuple[float, ...] | list[float] = HANDOFF_TOPK_PCTS,
    families: list[str] | None = None,
) -> int:
    n_cfgs = len(select_handoff_configs(topk_pcts=topk_pcts, families=families))
    n_tasks = n_longbench_tasks + n_benchmark_tasks
    return n_cfgs * num_examples * n_tasks
