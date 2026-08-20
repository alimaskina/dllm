"""YAML/dataclass configuration for sparse KV experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

BitSpec = Literal["fp16", "bf16", "8", "4", "2"]
SelectorMode = Literal["all_mean", "middle", "uniform"]
BaselineMode = Literal[
    "original",
    "dense_fp16",
    "dense_quant",
    "sparse_fp16",
    "sparse_quant_kv",
    "sparse_quant_kvq",
    "custom",
]


def parse_bits(spec: BitSpec | int | str | None) -> int:
    """Return effective bitwidth: 16 for fp16/bf16, else int bits."""
    if spec is None:
        return 16
    s = str(spec).lower()
    if s in ("fp16", "bf16", "16", "float16", "bfloat16"):
        return 16
    return int(s)


@dataclass
class SelectorConfig:
    mode: SelectorMode = "all_mean"
    uniform_n: int = 3
    topk: int = 256
    topk_pct: float | None = None
    per_head: bool = True

    def effective_topk(self, num_cached: int) -> int:
        """Fixed topk, or ceil pct of old cache when topk_pct is set."""
        if num_cached <= 0:
            return 0
        if self.topk_pct is not None:
            k = max(1, int(round(num_cached * self.topk_pct / 100.0)))
            return min(k, num_cached)
        return min(self.topk, num_cached)

    def validate(self) -> None:
        if self.mode not in ("all_mean", "middle", "uniform"):
            raise ValueError(f"unknown selector mode {self.mode!r}")
        if self.mode == "uniform" and self.uniform_n not in (3, 5, 7):
            raise ValueError(f"uniform_n must be 3, 5, or 7, got {self.uniform_n}")
        if self.topk_pct is not None:
            if not (0 < self.topk_pct <= 100):
                raise ValueError(f"topk_pct must be in (0, 100], got {self.topk_pct}")
        elif self.topk <= 0:
            raise ValueError("topk must be positive")


@dataclass
class PrecisionConfig:
    k_bits: BitSpec = "fp16"
    v_bits: BitSpec = "fp16"
    q_bits: BitSpec = "fp16"

    def as_ints(self) -> dict[str, int]:
        return {
            "k_bits": parse_bits(self.k_bits),
            "v_bits": parse_bits(self.v_bits),
            "q_bits": parse_bits(self.q_bits),
        }


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "smoke"
    baseline: BaselineMode = "custom"

    # Generation (Fast-dLLM-v2 defaults for GSM8K)
    model_path: str = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    block_size: int = 32
    small_block_size: int = 8
    threshold: float = 1.0
    max_new_tokens: int = 2048
    seed: int = 1234

    selector: SelectorConfig = field(default_factory=SelectorConfig)

    # Independent selector vs execution precision
    selector_precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    exec_precision: PrecisionConfig = field(default_factory=PrecisionConfig)

    # Sparse old-cache (False for dense baselines)
    sparse_old_cache: bool = True

    # KIVI geometry
    kivi_group_size: int = 32
    keys_pre_rope: bool = False

    # Eval
    num_examples: int = 3
    task: str = "gsm8k"

    # Logging
    save_full_cost_steps: bool = False
    log_selected_indices: bool = True

    # Output
    output_dir: str = "results/smoke"

    def validate(self) -> None:
        self.selector.validate()
        if self.block_size % self.small_block_size != 0:
            raise ValueError("block_size must be divisible by small_block_size")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        sel = raw.pop("selector", {}) or {}
        sel_prec = raw.pop("selector_precision", {}) or {}
        exec_prec = raw.pop("exec_precision", {}) or {}

        # Flat legacy keys
        for key in ("selector_k_bits", "selector_v_bits", "selector_q_bits"):
            if key in raw:
                attr = key.replace("selector_", "")
                sel_prec[attr] = raw.pop(key)
        for key in ("exec_k_bits", "exec_v_bits", "exec_q_bits"):
            if key in raw:
                attr = key.replace("exec_", "")
                exec_prec[attr] = raw.pop(key)

        cfg = cls(
            selector=SelectorConfig(**sel) if sel else SelectorConfig(),
            selector_precision=PrecisionConfig(**sel_prec) if sel_prec else PrecisionConfig(),
            exec_precision=PrecisionConfig(**exec_prec) if exec_prec else PrecisionConfig(),
            **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__},
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig.from_dict(raw)


def preset_config(name: str) -> ExperimentConfig:
    """Named smoke presets A–D."""
    base = ExperimentConfig(name=name, num_examples=3)
    presets: dict[str, ExperimentConfig] = {
        "A": ExperimentConfig(
            name="A_dense_fp16",
            baseline="original",
            sparse_old_cache=False,
            selector=SelectorConfig(topk=256),
            selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            exec_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            num_examples=3,
        ),
        "B": ExperimentConfig(
            name="B_sparse_fp16",
            baseline="sparse_fp16",
            sparse_old_cache=True,
            selector=SelectorConfig(topk=256),
            selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            exec_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            num_examples=3,
        ),
        "C": ExperimentConfig(
            name="C_sparse_quant_kv",
            baseline="sparse_quant_kv",
            sparse_old_cache=True,
            selector=SelectorConfig(topk=256),
            selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
            exec_precision=PrecisionConfig(k_bits="2", v_bits="2", q_bits="fp16"),
            num_examples=3,
        ),
        "D": ExperimentConfig(
            name="D_sparse_quant_kvq",
            baseline="sparse_quant_kvq",
            sparse_old_cache=True,
            selector=SelectorConfig(topk=256),
            selector_precision=PrecisionConfig(k_bits="2", v_bits="fp16", q_bits="4"),
            exec_precision=PrecisionConfig(k_bits="2", v_bits="2", q_bits="4"),
            num_examples=3,
        ),
    }
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}, expected one of {list(presets)}")
    return presets[name]
