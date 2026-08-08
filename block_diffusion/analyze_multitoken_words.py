#!/usr/bin/env python3
"""Multi-token word unmasking analysis for block-diffusion traces (traces.jsonl)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

# Reuse WikiText word segmentation + strict lexical filter + WordInstance types.
QWT = Path(__file__).resolve().parents[1] / "quant_where_to_unmask"
sys.path.insert(0, str(QWT))

from analyze_multitoken_words import (  # noqa: E402
    TokenUnmask,
    WordInstance,
    char_spans,
    find_word_instances_ws,
    order_pattern,
    print_report,
    step_gaps,
)
from multitoken_word_filters import filter_instances, is_lexical  # noqa: E402


@dataclass
class PosInfo:
    rel_gen_pos: int
    token_id: int
    global_step: int
    local_step: int
    block_idx: int
    block_offset: int
    preds_by_step: list[int]
    conf_by_step: list[float]


def _pos_lookup(trace: dict) -> dict[int, PosInfo]:
    out: dict[int, PosInfo] = {}
    offset = 0
    for block in trace["blocks"]:
        for p in block["positions"]:
            rel = int(p["rel_gen_pos"])
            preds = list(p.get("preds_by_step") or [])
            confs = list(p.get("conf_by_step") or [])
            if not preds and p.get("first_pred_id") is not None:
                preds = [int(p["first_pred_id"])]
            if not confs:
                confs = [None] * len(preds)  # type: ignore[list-item]
            out[rel] = PosInfo(
                rel_gen_pos=rel,
                token_id=int(p["final_pred_id"]),
                global_step=offset + int(p["unmask_step"]),
                local_step=int(p["unmask_step"]),
                block_idx=int(block["block_idx"]),
                block_offset=offset,
                preds_by_step=preds,
                conf_by_step=confs,
            )
        offset += int(block["inner_steps"])
    return out


def _pred_conf_at_step(info: PosInfo, local_step: int) -> tuple[int | None, float | None]:
    if not info.preds_by_step:
        return None, None
    idx = min(local_step, len(info.preds_by_step) - 1)
    conf = info.conf_by_step[idx] if idx < len(info.conf_by_step) else None
    return info.preds_by_step[idx], conf


def sibling_confidence_at_first_unmask(
    positions: list[int],
    lookup: dict[int, PosInfo],
    first_pos: int,
) -> list[dict]:
    first = lookup[first_pos]
    out = []
    for pos in positions:
        if pos == first_pos:
            continue
        sib = lookup.get(pos)
        if sib is None:
            continue
        if sib.block_idx != first.block_idx:
            # Word spans blocks: use sibling state at block entry.
            local_step = 0
        elif sib.local_step <= first.local_step:
            local_step = sib.local_step
        else:
            local_step = first.local_step
        pred_id, conf = _pred_conf_at_step(sib, local_step)
        out.append(
            {
                "pos_comp": pos,
                "still_masked": sib.global_step > first.global_step,
                "confidence": conf,
                "predicted_token_id": pred_id,
                "predicted_token": None,
            }
        )
    return out


def analyze_row(row: dict, tokenizer, *, kind_filter: str | None = "alpha") -> list[WordInstance]:
    trace = row["trace"]
    lookup = _pos_lookup(trace)
    if not lookup:
        return []

    n_gen = int(trace.get("n_gen_tokens") or row.get("n_gen_tokens") or (max(lookup) + 1))
    final_ids = [lookup[i].token_id if i in lookup else 0 for i in range(n_gen)]
    timeline = {
        pos: TokenUnmask(
            pos_comp=pos,
            token_id=info.token_id,
            token=tokenizer.decode([info.token_id]),
            step=info.global_step,
            confidence=info.conf_by_step[min(info.local_step, len(info.conf_by_step) - 1)]
            if info.conf_by_step
            else 0.0,
        )
        for pos, info in lookup.items()
    }

    spans, full = char_spans(final_ids, tokenizer)
    instances: list[WordInstance] = []
    for word, kind, positions in find_word_instances_ws(full, spans):
        core = re.sub(r"^[^\w']+|[^\w']+$", "", word)
        if len(core) < 2:
            continue
        if kind_filter and kind != kind_filter:
            continue
        tokens = [timeline[p] for p in positions if p in timeline]
        if len(tokens) < 2:
            continue
        first = min(tokens, key=lambda t: (t.step, t.pos_comp))
        sib = sibling_confidence_at_first_unmask(positions, lookup, first.pos_comp)
        for s in sib:
            if s["predicted_token_id"] is not None:
                s["predicted_token"] = tokenizer.decode([s["predicted_token_id"]])

        instances.append(
            WordInstance(
                word=word,
                kind=kind,
                positions=positions,
                tokens=tokens,
                consecutive=all(positions[i] + 1 == positions[i + 1] for i in range(len(positions) - 1)),
                same_step=len({t.step for t in tokens}) == 1,
                step_span=max(t.step for t in tokens) - min(t.step for t in tokens),
                first_pos=first.pos_comp,
                first_step=first.step,
                sibling_conf_at_first=sib,
            )
        )
    return instances


def load_traces(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_step_series(trace: dict) -> bool:
    for block in trace.get("blocks", []):
        for p in block.get("positions", []):
            if p.get("preds_by_step"):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, help="traces.jsonl from run_volatility.py")
    parser.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument(
        "--word-tier",
        choices=["none", "alpha", "lexical", "lexical_loose"],
        default="lexical",
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    path = Path(args.traces)
    rows = load_traces(path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_instances: list[WordInstance] = []
    with_series = 0
    for row in rows:
        if has_step_series(row["trace"]):
            with_series += 1
        all_instances.extend(analyze_row(row, tokenizer, kind_filter="alpha"))
    all_instances = filter_instances(all_instances, tier=args.word_tier)

    if with_series < len(rows):
        print(
            f"WARNING: {len(rows) - with_series}/{len(rows)} traces lack preds_by_step/conf_by_step. "
            "Re-run run_volatility.py for full sibling metrics."
        )

    print(f"traces={len(rows)}  lexical_words={len(all_instances)}  tier={args.word_tier}")
    if args.report:
        print_report(
            all_instances,
            n_traces=len(rows),
            checkpoint=path.parent,
            method="ws",
            kind=f"alpha/{args.word_tier}",
        )


if __name__ == "__main__":
    main()
