"""Three LongBench experiment presets."""

from __future__ import annotations

from config import ExperimentConfig, PrecisionConfig, SelectorConfig


def longbench_presets() -> dict[str, ExperimentConfig]:
    common = dict(
        block_size=32,
        small_block_size=8,
        threshold=1.0,
        max_new_tokens=2048,  # overridden per-task at runtime
        log_selected_indices=False,
        save_full_cost_steps=True,
    )

    return {
        "baseline": ExperimentConfig(
            name="baseline_dense_fp16",
            baseline="original",
            sparse_old_cache=False,
            **common,
        ),
        "extreme_k128_k2v2": ExperimentConfig(
            name="extreme_k128_all_mean_k2v2",
            baseline="sparse_quant_kv",
            sparse_old_cache=True,
            selector=SelectorConfig(mode="all_mean", topk=128),
            selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            exec_precision=PrecisionConfig(k_bits="2", v_bits="2", q_bits="fp16"),
            **common,
        ),
        "middle_fp16": ExperimentConfig(
            name="sparse_k128_middle_fp16",
            baseline="sparse_fp16",
            sparse_old_cache=True,
            selector=SelectorConfig(mode="middle", topk=128),
            selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            exec_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            **common,
        ),
    }
