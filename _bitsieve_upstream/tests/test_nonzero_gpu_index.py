#!/usr/bin/env python3
"""Regression test for a real, deterministic crash: any CUDA index other than
0, used without CUDA_VISIBLE_DEVICES restricting visibility to it, fails
every single generate() call with

    ValueError: Pointer argument (at 0) cannot be accessed from Triton
    (cpu tensor?)

- not because a tensor is on CPU, but because torch.cuda.current_device()
stays 0 (creating a tensor on cuda:N via device= or .to() does not move it),
so the packed/selector Triton kernels launch against device 0's context while
their tensors live on device N. load_fast_dllm() now calls
torch.cuda.set_device() before anything else touches CUDA, which fixes this
for every entry point that loads the model through it (run_suite.py,
eval.quality, eval.performance, ...).

Needs a machine with >= 2 visible GPUs and none of them pinned via
CUDA_VISIBLE_DEVICES (skips otherwise, rather than passing vacuously).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import load_benchmark  # noqa: E402
from bitsieve_fastdllm.eval.common import encode_prompt, load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.runtime.generator import BitSieveGenerator  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("SKIP: needs >= 2 visible GPUs (this test is meaningless under "
              "CUDA_VISIBLE_DEVICES restricted to one device)")
        return 0

    target = "cuda:1"
    print(f"current_device before load_fast_dllm: {torch.cuda.current_device()}")
    model, tokenizer = load_fast_dllm(
        "Efficient-Large-Model/Fast_dLLM_v2_7B", dtype=torch.bfloat16, device=target
    )
    after = torch.cuda.current_device()
    print(f"current_device after load_fast_dllm: {after}")
    ok = after == 1
    print(f"  [{'PASS' if ok else 'FAIL'}] torch.cuda.current_device() == 1 after "
          f"load_fast_dllm(device='cuda:1')")
    if not ok:
        return 1

    examples = load_benchmark("gsm8k", tokenizer=tokenizer, limit=1, split="test")
    cfg = ExperimentConfig.load(ROOT / "configs" / "suite" / "sparse_fp16_all_p5.yaml")
    ids = encode_prompt(
        tokenizer, examples[0].prompt,
        max_input_tokens=cfg.max_cache_tokens - cfg.generation.max_new_tokens,
        use_chat_template=True, device=target,
    )
    gen = BitSieveGenerator(model, tokenizer, cfg)
    result = gen.generate(ids)
    ok = bool(result.texts[0].strip())
    print(f"  [{'PASS' if ok else 'FAIL'}] a sparse selector generate() call on cuda:1 "
          f"produced output ({result.texts[0][:60]!r}...)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
