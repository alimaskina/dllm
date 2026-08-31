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
    # Top-k over contiguous cache-token groups (1 = per-token, legacy behavior).
    token_group_size: int = 1
    token_group_reduce: Literal["sum", "max"] = "sum"

    def effective_topk(self, num_cached: int) -> int:
        """Fixed topk, or ceil pct of old cache when topk_pct is set."""
        if num_cached <= 0:
            return 0
        if self.topk_pct is not None:
            k = max(1, int(round(num_cached * self.topk_pct / 100.0)))
            return min(k, num_cached)
        return min(self.topk, num_cached)

    def effective_topk_groups(self, num_cached: int) -> int:
        """Whole token groups to pick when token_group_size > 1."""
        if num_cached <= 0:
            return 0
        k = self.effective_topk(num_cached)
        g = max(1, self.token_group_size)
        n_groups = (num_cached + g - 1) // g
        return min(n_groups, max(1, (k + g - 1) // g))

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
        if self.token_group_size <= 0:
            raise ValueError("token_group_size must be positive")
        if self.token_group_reduce not in ("sum", "max"):
            raise ValueError(f"token_group_reduce must be sum or max, got {self.token_group_reduce!r}")


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

    # KIVI geometry (token groups for K; see kv_quant/kv_cache_quant.py)
    kivi_group_size: int = 32
    kivi_residual_length: int = 32
    keys_pre_rope: bool = False
    # Key quant for probe + exec views: kivi | per_token | hadamard_per_token | quarot | qjl
    k_quant_scheme: str = "kivi"
    # Value quant: kivi (per-token min/max) | hadamard_per_token
    v_quant_scheme: str = "kivi"
    # If set (e.g. "4") *and* selector_precision is FP16: rank via a quant-K
    # probe. Prefer selector_precision.k_bits instead — first-pass SDPA uses
    # that K, and top-k is taken from the captured first-pass map.
    rank_k_bits: str | None = None

    # Eval
    num_examples: int = 3
    task: str = "gsm8k"

    # Logging
    save_full_cost_steps: bool = False
    log_selected_indices: bool = True

    # Output
    output_dir: str = "results/smoke"

    # Optional: after selecting top-k, requantize the sparse exec tensors again.
    # This simulates storing selected tokens in low-bit form separately from the
    # full-cache view (still fake-quant: quant→dequant back to float).
    requantize_sparse_exec: bool = False
    # When requantizing sparse exec, which source view to pull tokens from.
    # - "exec": take tokens from exec view (e.g. fp16→k4), then requantize (k4→k4).
    # - "selector": take tokens from selector view (e.g. fp16→k2), then requantize
    #   into exec bits (k2→k4). This removes any “direct fp16→k4” advantage for exec.
    requantize_sparse_exec_source: str = "exec"
    # Control whether sparse-exec requantization applies to K and/or V.
    # None means "follow requantize_sparse_exec".
    requantize_sparse_exec_k: bool | None = None
    requantize_sparse_exec_v: bool | None = None

    def validate(self) -> None:
        self.selector.validate()
        if self.block_size % self.small_block_size != 0:
            raise ValueError("block_size must be divisible by small_block_size")
        if self.k_quant_scheme not in (
            "kivi",
            "per_token",
            "hadamard_per_token",
            "quarot",
            "qjl",
        ):
            raise ValueError(
                f"k_quant_scheme must be kivi, per_token, hadamard_per_token, quarot, or qjl, "
                f"got {self.k_quant_scheme!r}"
            )
        if self.v_quant_scheme not in ("kivi", "hadamard_per_token"):
            raise ValueError(
                f"v_quant_scheme must be kivi or hadamard_per_token, got {self.v_quant_scheme!r}"
            )
        if self.requantize_sparse_exec_source not in ("exec", "selector"):
            raise ValueError(
                "requantize_sparse_exec_source must be 'exec' or 'selector', "
                f"got {self.requantize_sparse_exec_source!r}"
            )

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
