#!/usr/bin/env python3
"""Config invariants that keep the measurements honest. No GPU needed.

Each assertion here corresponds to a way an earlier revision could report a
number that described something other than the method being claimed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import max_new_tokens_for  # noqa: E402

CONFIG_DIR = ROOT / "configs"


def main() -> int:
    failures: list[str] = []

    def chk(ok: bool, msg: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            failures.append(msg)

    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    chk(bool(paths), f"found configs ({len(paths)})")

    print("\n=== no full-precision residual, no dense-exempt layers ===")
    for path in paths:
        cfg = ExperimentConfig.load(path)
        chk(
            cfg.quant.residual_tokens == 0,
            f"{path.name}: residual_tokens == 0 (got {cfg.quant.residual_tokens})",
        )
        chk(
            cfg.selector.dense_prefix_layers == 0,
            f"{path.name}: dense_prefix_layers == 0 "
            f"(got {cfg.selector.dense_prefix_layers})",
        )

    print("\n=== the key group divides the block, so the residual stays empty ===")
    for path in paths:
        cfg = ExperimentConfig.load(path)
        block = cfg.generation.block_size
        group = cfg.quant.key_token_group
        chk(
            block % group == 0 or group % block == 0,
            f"{path.name}: block_size {block} and key_token_group {group} are "
            "commensurate (otherwise a partial group would force a float tail)",
        )

    print("\n=== math gets room for a full chain of thought ===")
    for bench in ("gsm8k", "math500"):
        chk(max_new_tokens_for(bench) == 2048, f"{bench} budget == 2048")
    chk(max_new_tokens_for("niah") == 64, "niah budget == 64")
    chk(max_new_tokens_for("qmsum") == 512, "qmsum budget == 512")

    print("\n=== a percentage budget engages selection at any prompt length ===")
    for name in ("proposed_a_k4v4_p5", "proposed_a_k2v2_p5"):
        cfg = ExperimentConfig.load(CONFIG_DIR / f"{name}.yaml")
        chk(cfg.selector.topk_percent is not None, f"{name}: topk_percent is set")
        # The bypass rule is `effective_topk >= prefix_len`; a percentage below
        # 100 can never reach it, so selection always has something to discard.
        short, long = 64, 20000
        chk(
            cfg.selector.effective_topk(short) < short
            and cfg.selector.effective_topk(long) < long,
            f"{name}: budget stays under the prefix at {short} and {long} tokens",
        )

    print("\n=== a fixed budget is honest about not engaging ===")
    cfg = ExperimentConfig.load(CONFIG_DIR / "proposed_a_k4v4_k512.yaml")
    chk(
        cfg.selector.effective_topk(400) == 400,
        "k512 at a 400-token prefix selects everything, i.e. runs dense - this "
        "is why sparse_block_fraction must be reported",
    )

    print("\n=== coverage scoring is refused where it would be meaningless ===")
    raw = ExperimentConfig.load(CONFIG_DIR / "official_dense_bf16.yaml").to_dict()
    raw["coverage_diagnostics"] = True
    try:
        ExperimentConfig.from_dict(raw)
        chk(False, "dense engine + coverage_diagnostics is rejected")
    except ValueError:
        chk(True, "dense engine + coverage_diagnostics is rejected")

    print(f"\n{'ALL PASS' if not failures else str(len(failures)) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
