from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..runtime.generator import BitSieveGenerator
from ..runtime.official_generator import OfficialDenseGenerator
from .common import load_fast_dllm, parse_dtype


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _build_exact_length_prompt(
    tokenizer,
    length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    text = (
        "This is a controlled long-context systems benchmark for block diffusion language model "
        "inference. The content is intentionally repetitive and semantically harmless. "
    )
    base = tokenizer.encode(text, add_special_tokens=False)
    if not base:
        raise RuntimeError("tokenizer produced an empty sequence")
    tokens = (base * ((length + len(base) - 1) // len(base)))[:length]
    row = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    return row.expand(batch_size, -1).contiguous()


def _runtime_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    raw = cfg.to_dict()
    generation = raw["generation"]
    generation["max_new_tokens"] = int(args.max_new_tokens)
    generation["stop_token_id"] = None
    if args.schedule == "fixed":
        generation["schedule"] = "fixed"
        generation["fixed_steps_per_block"] = int(args.fixed_steps)
        generation["small_block_size"] = generation["block_size"]
    else:
        generation["schedule"] = "threshold"
    if args.max_cache_tokens is not None:
        raw["max_cache_tokens"] = int(args.max_cache_tokens)
    return ExperimentConfig.from_dict(raw)


def _median(measurements: list[dict[str, Any]], name: str) -> float | None:
    vals = [float(x[name]) for x in measurements if x.get(name) is not None]
    return statistics.median(vals) if vals else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='End-to-end Fast-dLLM latency, TPOB, throughput, and memory benchmark.')
    p.add_argument("--config", action="append", required=True, help="repeat for each method")
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--revision")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-implementation")
    p.add_argument("--contexts", default="2048,8192,16384,28672")
    p.add_argument("--batch-sizes", default="1,4")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-cache-tokens", type=int)
    p.add_argument("--schedule", choices=("fixed", "threshold"), default="fixed")
    p.add_argument("--fixed-steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.device != "auto" and not torch.cuda.is_available():
        raise RuntimeError("performance evaluation requires a CUDA GPU")
    base_configs = [ExperimentConfig.load(path) for path in args.config]
    configs = [_runtime_config(cfg, args) for cfg in base_configs]
    model, tokenizer = load_fast_dllm(
        args.model,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
    )
    device = next(model.parameters()).device
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")

    for cfg in configs:
        generator = (
            BitSieveGenerator(model, tokenizer, cfg)
            if cfg.engine == "bitsieve"
            else OfficialDenseGenerator(model, tokenizer, cfg)
        )
        for batch_size in _ints(args.batch_sizes):
            for context in _ints(args.contexts):
                if context + cfg.generation.max_new_tokens > cfg.max_cache_tokens:
                    print(
                        f"skip {cfg.name} batch={batch_size} context={context}: "
                        f"capacity {cfg.max_cache_tokens} is too small"
                    )
                    continue
                input_ids = _build_exact_length_prompt(
                    tokenizer, context, batch_size, device
                )
                for _ in range(args.warmup):
                    generator.generate(input_ids)
                torch.cuda.synchronize(device)

                measurements: list[dict[str, Any]] = []
                for repeat in range(args.repeats):
                    result = generator.generate(input_ids)
                    metrics = result.metrics
                    measurements.append(metrics)
                    row = {
                        "config_name": cfg.name,
                        "config": cfg.to_dict(),
                        "model_id": args.model,
                        "model_revision": args.revision,
                        "dtype": args.dtype,
                        "device_name": torch.cuda.get_device_name(device),
                        "context_tokens_per_request": context,
                        "batch_size": batch_size,
                        "repeat": repeat,
                        **metrics,
                    }
                    with out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

                peak = _median(measurements, "peak_cuda_allocated_bytes")
                print(
                    json.dumps(
                        {
                            "config": cfg.name,
                            "batch_size": batch_size,
                            "context": context,
                            "median_prefill_ms": _median(measurements, "prefill_ms"),
                            "median_decode_ms": _median(measurements, "decode_ms"),
                            "median_tpob_ms": _median(measurements, "mean_tpob_ms"),
                            "median_tokens_per_second": _median(
                                measurements, "tokens_per_second"
                            ),
                            "median_peak_allocated_gib": peak / 2**30 if peak else None,
                        },
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
