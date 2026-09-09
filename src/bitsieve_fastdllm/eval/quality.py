from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

from ..config import ExperimentConfig
from ..runtime.generator import BitSieveGenerator
from ..runtime.official_generator import OfficialDenseGenerator
from .benchmarks import load_benchmark
from .common import encode_prompt, load_fast_dllm, parse_dtype
from .metrics import score_prediction
from .resume import load_resume_rows, rewrite_existing_rows


def _parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run quality benchmarks with the official dense or BitSieve engine.')
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--revision")
    p.add_argument("--attn-implementation")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--max-cache-tokens", type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--plain-prompt", action="store_true")
    p.add_argument("--niah-contexts", default="8192,16384,28672")
    p.add_argument("--niah-depths", default="0,0.25,0.5,0.75,1")
    p.add_argument("--save-traces", action="store_true")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "resume from OUTPUT or OUTPUT.partial when compatible rows exist "
            "(default: enabled; use --no-resume for a clean overwrite)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = ExperimentConfig.load(args.config)
    if args.max_new_tokens is not None or args.max_cache_tokens is not None:
        raw = cfg.to_dict()
        if args.max_new_tokens is not None:
            raw["generation"]["max_new_tokens"] = args.max_new_tokens
        if args.max_cache_tokens is not None:
            raw["max_cache_tokens"] = args.max_cache_tokens
        cfg = ExperimentConfig.from_dict(raw)
    model, tokenizer = load_fast_dllm(
        args.model,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
    )
    examples = load_benchmark(
        args.benchmark,
        tokenizer=tokenizer,
        limit=args.limit,
        split=args.split,
        niah_contexts=_parse_ints(args.niah_contexts),
        niah_depths=_parse_floats(args.niah_depths),
        seed=cfg.seed,
    )
    generator = (
        BitSieveGenerator(model, tokenizer, cfg)
        if cfg.engine == "bitsieve"
        else OfficialDenseGenerator(model, tokenizer, cfg)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = output_path.parent / f"{output_path.stem}_traces"

    example_ids = [str(example.example_id) for example in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("benchmark contains duplicate example ids; safe resume is impossible")

    resume_sources: list[Path] = []
    if args.resume:
        rows_by_id, resume_sources = load_resume_rows(
            output_path,
            benchmark=args.benchmark,
            config=cfg,
            model_id=args.model,
            model_revision=args.revision,
            dtype=args.dtype,
            valid_ids=set(example_ids),
        )
        rows = rewrite_existing_rows(output_path, examples, rows_by_id)
    else:
        rows = []

    completed_ids = {str(row["id"]) for row in rows}
    total = sum(float(row["score"]) for row in rows)
    if rows:
        print(
            f"resuming {args.benchmark}:{cfg.name} from {len(rows)}/{len(examples)} examples "
            f"using {', '.join(str(path) for path in resume_sources)}",
            file=sys.stderr,
            flush=True,
        )

    mode = "a" if rows else "w"
    with output_path.open(mode, encoding="utf-8") as handle, tqdm(
        total=len(examples),
        initial=len(rows),
        desc=f"{args.benchmark}:{cfg.name}",
    ) as progress:
        for example in examples:
            if str(example.example_id) in completed_ids:
                continue
            max_input = cfg.max_cache_tokens - cfg.generation.max_new_tokens
            if max_input <= 0:
                raise ValueError("max_cache_tokens must exceed max_new_tokens")
            input_ids = encode_prompt(
                tokenizer,
                example.prompt,
                max_input_tokens=max_input,
                use_chat_template=not args.plain_prompt,
                device=next(model.parameters()).device,
            )
            result = generator.generate(input_ids)
            prediction = result.texts[0]
            metrics = result.metrics
            trace = result.trace
            score = score_prediction(args.benchmark, prediction, example.references)
            total += score
            row = {
                "id": example.example_id,
                "benchmark": args.benchmark,
                "config": cfg.to_dict(),
                "model_id": args.model,
                "model_revision": args.revision,
                "dtype": args.dtype,
                "prompt_tokens": int(input_ids.shape[1]),
                "prediction": prediction,
                "references": example.references,
                "score": score,
                "metadata": example.metadata,
                "runtime": metrics,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            completed_ids.add(str(example.example_id))
            progress.update(1)
            if args.save_traces and trace is not None:
                trace.dump(trace_dir / f"{example.example_id}.json")

    summary = {
        "benchmark": args.benchmark,
        "config": cfg.to_dict(),
        "model_id": args.model,
        "model_revision": args.revision,
        "dtype": args.dtype,
        "num_examples": len(rows),
        "mean_score": total / len(rows) if rows else None,
        "mean_decode_ms": (
            sum(float(x["runtime"].get("decode_ms", 0.0)) for x in rows) / len(rows)
            if rows
            else None
        ),
        "mean_tokens_per_second": (
            sum(
                float(x["runtime"].get("tokens_per_second") or 0.0)
                for x in rows
            )
            / len(rows)
            if rows
            else None
        ),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for source in resume_sources:
        if source != output_path:
            source.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
