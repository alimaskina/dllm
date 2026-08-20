"""LongBench sweep grid: fixed topk and/or cache-percent topk × {baseline, middle, extreme}."""

from __future__ import annotations

from config import ExperimentConfig, PrecisionConfig, SelectorConfig

TOPKS = (64, 128, 256)
TOPK_PCTS = (2.5, 5, 10, 20)
EXTREME_QUANT = (("2", "2"), ("4", "2"), ("2", "4"), ("4", "4"))

_COMMON = dict(
    block_size=32,
    small_block_size=8,
    threshold=1.0,
    max_new_tokens=2048,
    log_selected_indices=False,
    save_full_cost_steps=True,
)


def pct_to_label(p: float) -> str:
    """2.5 → p2p5, 5 → p5."""
    return "p" + f"{p:g}".replace(".", "p")


def label_to_pct(label: str) -> float:
    """p2p5 → 2.5, p5 → 5."""
    body = label[1:] if label.startswith("p") else label
    if "p" in body:
        whole, frac = body.split("p", 1)
        return float(f"{whole}.{frac}")
    return float(body)


def _sparse(
    *,
    name: str,
    mode: str,
    exec_k: str,
    exec_v: str,
    topk: int | None = None,
    topk_pct: float | None = None,
    uniform_n: int = 3,
    baseline: str = "sparse_quant_kv",
) -> ExperimentConfig:
    if topk_pct is None and topk is None:
        raise ValueError("need topk or topk_pct")
    if exec_k == "fp16" and exec_v == "fp16":
        baseline = "sparse_fp16"
    elif exec_k != "fp16" or exec_v != "fp16":
        baseline = "sparse_quant_kv"
    sel_kw: dict = dict(mode=mode, uniform_n=uniform_n)  # type: ignore[arg-type]
    if topk_pct is not None:
        sel_kw["topk_pct"] = topk_pct
        sel_kw["topk"] = 256
    else:
        sel_kw["topk"] = topk
    return ExperimentConfig(
        name=name,
        baseline=baseline,  # type: ignore[arg-type]
        sparse_old_cache=True,
        selector=SelectorConfig(**sel_kw),
        selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
        exec_precision=PrecisionConfig(k_bits=exec_k, v_bits=exec_v, q_bits="fp16"),  # type: ignore[arg-type]
        **_COMMON,
    )


def _add_fixed_k_configs(out: dict[str, ExperimentConfig], topks: tuple[int, ...] | list[int]) -> None:
    for k in topks:
        out[f"middle_k{k}"] = _sparse(
            name=f"middle_k{k}_fp16",
            mode="middle",
            topk=k,
            exec_k="fp16",
            exec_v="fp16",
        )
        for ek, ev in EXTREME_QUANT:
            key = f"extreme_k{k}_k{ek}v{ev}"
            out[key] = _sparse(
                name=f"extreme_k{k}_all_mean_k{ek}v{ev}",
                mode="all_mean",
                topk=k,
                exec_k=ek,
                exec_v=ev,
            )


def _add_pct_configs(out: dict[str, ExperimentConfig], topk_pcts: tuple[float, ...] | list[float]) -> None:
    for pct in topk_pcts:
        pl = pct_to_label(pct)
        out[f"middle_{pl}"] = _sparse(
            name=f"middle_{pl}_fp16",
            mode="middle",
            topk_pct=pct,
            exec_k="fp16",
            exec_v="fp16",
        )
        for ek, ev in EXTREME_QUANT:
            key = f"extreme_{pl}_k{ek}v{ev}"
            out[key] = _sparse(
                name=f"extreme_{pl}_all_mean_k{ek}v{ev}",
                mode="all_mean",
                topk_pct=pct,
                exec_k=ek,
                exec_v=ev,
            )


def longbench_sweep_grid(topks: tuple[int, ...] | list[int] = TOPKS) -> dict[str, ExperimentConfig]:
    out: dict[str, ExperimentConfig] = {
        "baseline": ExperimentConfig(
            name="baseline_dense_fp16",
            baseline="original",
            sparse_old_cache=False,
            **_COMMON,
        ),
    }
    _add_fixed_k_configs(out, topks)
    return out


def longbench_pct_sweep_grid(
    topk_pcts: tuple[float, ...] | list[float] = TOPK_PCTS,
) -> dict[str, ExperimentConfig]:
    out: dict[str, ExperimentConfig] = {
        "baseline": ExperimentConfig(
            name="baseline_dense_fp16",
            baseline="original",
            sparse_old_cache=False,
            **_COMMON,
        ),
    }
    _add_pct_configs(out, topk_pcts)
    return out


def combined_sweep_grid(
    topks: tuple[int, ...] | list[int] = (),
    topk_pcts: tuple[float, ...] | list[float] = (),
) -> dict[str, ExperimentConfig]:
    out: dict[str, ExperimentConfig] = {
        "baseline": ExperimentConfig(
            name="baseline_dense_fp16",
            baseline="original",
            sparse_old_cache=False,
            **_COMMON,
        ),
    }
    if topks:
        _add_fixed_k_configs(out, topks)
    if topk_pcts:
        _add_pct_configs(out, topk_pcts)
    return out


def sweep_config_list(
    topks: tuple[int, ...] | list[int] = TOPKS,
    topk_pcts: tuple[float, ...] | list[float] = (),
) -> list[ExperimentConfig]:
    """Ordered: baseline, then each topk / topk_pct → middle → k2v2 → k4v2 → k4v4."""
    grid = combined_sweep_grid(topks, topk_pcts)
    order = ["baseline"]
    for k in topks:
        order.append(f"middle_k{k}")
        for ek, ev in EXTREME_QUANT:
            order.append(f"extreme_k{k}_k{ek}v{ev}")
    for pct in topk_pcts:
        pl = pct_to_label(pct)
        order.append(f"middle_{pl}")
        for ek, ev in EXTREME_QUANT:
            order.append(f"extreme_{pl}_k{ek}v{ev}")
    return [grid[k] for k in order if k in grid]


def select_sweep_configs(
    *,
    topks: tuple[int, ...] | list[int] = TOPKS,
    topk_pcts: tuple[float, ...] | list[float] = (),
    families: list[str],
) -> list[ExperimentConfig]:
    grid = combined_sweep_grid(topks, topk_pcts)
    selected: list[ExperimentConfig] = []

    def maybe_add(cfg: ExperimentConfig) -> None:
        if cfg.name not in {c.name for c in selected}:
            selected.append(cfg)

    if "baseline" in families and "baseline" in grid:
        maybe_add(grid["baseline"])

    for k in topks:
        if f"middle_k{k}" in grid and "middle" in families:
            maybe_add(grid[f"middle_k{k}"])
        for fam, ek, ev in [
            ("extreme_k2v2", "2", "2"),
            ("extreme_k4v2", "4", "2"),
            ("extreme_k2v4", "2", "4"),
            ("extreme_k4v4", "4", "4"),
        ]:
            key = f"extreme_k{k}_k{ek}v{ev}"
            if fam in families and key in grid:
                maybe_add(grid[key])

    for pct in topk_pcts:
        pl = pct_to_label(pct)
        if f"middle_{pl}" in grid and "middle" in families:
            maybe_add(grid[f"middle_{pl}"])
        for fam, ek, ev in [
            ("extreme_k2v2", "2", "2"),
            ("extreme_k4v2", "4", "2"),
            ("extreme_k2v4", "2", "4"),
            ("extreme_k4v4", "4", "4"),
        ]:
            key = f"extreme_{pl}_k{ek}v{ev}"
            if fam in families and key in grid:
                maybe_add(grid[key])

    return selected
