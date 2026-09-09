from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Semantic = Literal["dense", "A", "B"]
SelectorMode = Literal["all", "middle", "uniform"]
SelectorDomain = Literal["prefix", "full"]
SelectorScore = Literal["softmax", "raw"]
Backend = Literal["auto", "triton", "torch"]
KernelVariant = Literal["blocked", "legacy"]
SelectorLogitsDtype = Literal["float16", "float32"]
Engine = Literal["bitsieve", "official"]
CompactFormat = Literal["bf16", "requantized"]
UnmaskSchedule = Literal["threshold", "fixed"]


def _validate_bits(name: str, bits: int) -> None:
    if bits not in (2, 4, 16):
        raise ValueError(f"{name} must be 2, 4, or 16; got {bits}")


@dataclass(slots=True)
class QuantizationConfig:

    k_bits: int = 4
    v_bits: int = 4
    key_token_group: int = 32
    value_channel_group: int = 32
    residual_tokens: int = 32
    param_dtype: Literal["float16", "bfloat16", "float32"] = "float16"

    def validate(self, *, head_dim: int | None = None) -> None:
        _validate_bits("k_bits", self.k_bits)
        _validate_bits("v_bits", self.v_bits)
        if self.key_token_group <= 0 or self.key_token_group % 8 != 0:
            raise ValueError("key_token_group must be a positive multiple of 8")
        if self.value_channel_group <= 0 or self.value_channel_group % 8 != 0:
            raise ValueError("value_channel_group must be a positive multiple of 8")
        if self.residual_tokens < 0:
            raise ValueError("residual_tokens must be non-negative")
        if head_dim is not None and head_dim % self.value_channel_group:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by value_channel_group="
                f"{self.value_channel_group}"
            )
        if self.k_bits < 16 and self.key_token_group % (8 // self.k_bits):
            raise ValueError("key_token_group is incompatible with packed key bit width")
        if self.v_bits < 16 and self.value_channel_group % (8 // self.v_bits):
            raise ValueError("value_channel_group is incompatible with packed value bit width")


@dataclass(slots=True)
class SelectorConfig:

    mode: SelectorMode = "uniform"
    uniform_queries: int = 5
    topk: int = 512
    topk_percent: float | None = None
    domain: SelectorDomain = "prefix"
    score: SelectorScore = "softmax"
    dense_prefix_layers: int = 2
    sort_indices: bool = True

    def validate(self, *, block_size: int | None = None, num_layers: int | None = None) -> None:
        if self.mode not in ("all", "middle", "uniform"):
            raise ValueError(f"unknown selector mode: {self.mode}")
        if self.uniform_queries <= 0:
            raise ValueError("uniform_queries must be positive")
        if block_size is not None and self.uniform_queries > block_size:
            raise ValueError("uniform_queries cannot exceed block_size")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.topk_percent is not None and not (0.0 < self.topk_percent <= 100.0):
            raise ValueError("topk_percent must be in (0, 100]")
        if self.dense_prefix_layers < 0:
            raise ValueError("dense_prefix_layers must be non-negative")
        if num_layers is not None and self.dense_prefix_layers > num_layers:
            raise ValueError("dense_prefix_layers exceeds the number of layers")

    def effective_topk(self, old_cache_len: int) -> int:
        if old_cache_len <= 0:
            return 0
        if self.topk_percent is not None:
            return min(
                old_cache_len,
                max(1, int(math.ceil(old_cache_len * self.topk_percent / 100.0))),
            )
        return min(old_cache_len, self.topk)

    def query_indices(self, masked_positions: list[int], block_size: int) -> list[int]:
        positions = sorted(set(int(x) for x in masked_positions if 0 <= int(x) < block_size))
        if not positions:
            return []
        if self.mode == "all":
            return positions
        if self.mode == "middle":
            center = (block_size - 1) / 2.0
            return [min(positions, key=lambda x: (abs(x - center), x))]

        n = min(self.uniform_queries, len(positions))
        if n == 1:
            return [positions[len(positions) // 2]]


        chosen = {
            positions[int(round(i * (len(positions) - 1) / (n - 1)))] for i in range(n)
        }
        return sorted(chosen)


@dataclass(slots=True)
class GenerationConfig:
    block_size: int = 32
    small_block_size: int = 8
    threshold: float = 0.95
    max_new_tokens: int = 512
    mask_token_id: int = 151665
    stop_token_id: int | None = 151645
    top_p: float = 0.95
    temperature: float = 0.0
    use_block_cache: bool = True
    schedule: UnmaskSchedule = "threshold"
    fixed_steps_per_block: int = 20

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.small_block_size <= 0 or self.block_size % self.small_block_size:
            raise ValueError("small_block_size must be a positive divisor of block_size")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.schedule == "fixed":
            if self.fixed_steps_per_block <= 0:
                raise ValueError("fixed_steps_per_block must be positive")
            if self.small_block_size != self.block_size:
                raise ValueError(
                    "fixed schedule requires small_block_size == block_size for controlled TPOB"
                )


@dataclass(slots=True)
class ExperimentConfig:

    name: str = "proposed_a_uniform5_k4v4"
    semantic: Semantic = "A"
    quant: QuantizationConfig = field(default_factory=QuantizationConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    engine: Engine = "bitsieve"
    backend: Backend = "auto"
    async_selector: bool = True
    max_cache_tokens: int = 32768
    compact_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    compact_format: CompactFormat = "bf16"
    seed: int = 1234
    collect_diagnostics: bool = False
    # Score each selection against the TRUE fp16 attention it was meant to
    # approximate. Keeps a shadow fp16 key cache and recomputes the reference
    # ranking from ALL masked block queries, independent of the query subset or
    # key precision the config under test actually used - so a starved or
    # quantized selector cannot grade its own homework. Diagnostic only:
    # costs memory and time, so it is off by default and never on in
    # performance or memory runs.
    coverage_diagnostics: bool = False
    # Query rows per chunk when recomputing the fp16 reference (caps the
    # transient [rows x prefix] logit tensor on long prefixes).
    coverage_query_chunk: int = 32
    profile_layers: bool = False
    dense_kernel_variant: KernelVariant = "blocked"
    selector_kernel_variant: KernelVariant = "blocked"
    selector_logits_dtype: SelectorLogitsDtype = "float16"
    validate_attention_masks: bool = False

    def validate(
        self,
        *,
        head_dim: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        if self.semantic not in ("dense", "A", "B"):
            raise ValueError(f"unknown semantic: {self.semantic}")
        if self.engine not in ("bitsieve", "official"):
            raise ValueError(f"unknown engine: {self.engine}")
        if self.engine == "official" and self.semantic != "dense":
            raise ValueError("the official engine is only valid for the dense baseline")
        self.generation.validate()
        self.quant.validate(head_dim=head_dim)
        self.selector.validate(
            block_size=self.generation.block_size,
            num_layers=num_layers,
        )
        if self.dense_kernel_variant not in ("blocked", "legacy"):
            raise ValueError(f"unknown dense_kernel_variant: {self.dense_kernel_variant}")
        if self.selector_kernel_variant not in ("blocked", "legacy"):
            raise ValueError(f"unknown selector_kernel_variant: {self.selector_kernel_variant}")
        if self.selector_logits_dtype not in ("float16", "float32"):
            raise ValueError(f"unknown selector_logits_dtype: {self.selector_logits_dtype}")
        if self.compact_format not in ("bf16", "requantized"):
            raise ValueError(f"unknown compact_format: {self.compact_format}")
        if self.compact_format == "requantized" and self.engine != "bitsieve":
            raise ValueError("requantized compact cache requires the BitSieve engine")
        if self.max_cache_tokens < self.generation.block_size:
            raise ValueError("max_cache_tokens is smaller than one generation block")
        if self.coverage_query_chunk <= 0:
            raise ValueError("coverage_query_chunk must be positive")
        if self.coverage_diagnostics and self.semantic == "dense":
            raise ValueError(
                "coverage_diagnostics needs a selector to score; it is meaningless "
                "for the dense baseline"
            )
        if self.semantic == "dense" and self.async_selector:

            self.async_selector = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        data = dict(raw)
        quant = QuantizationConfig(**data.pop("quant", {}))
        selector = SelectorConfig(**data.pop("selector", {}))
        generation = GenerationConfig(**data.pop("generation", {}))
        cfg = cls(quant=quant, selector=selector, generation=generation, **data)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"configuration in {p} must be a mapping")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        else:
            p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
