"""
Instrumented block-diffusion generation for Fast-dLLM v2.

Tracks argmax predictions at each inner forward pass within a block and
compares them to the first-pass prediction for each masked position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645


@dataclass
class PositionTrace:
    abs_pos: int
    rel_gen_pos: int
    pos_in_block: int
    first_pred: int | None = None
    final_pred: int | None = None
    preds_by_step: list[int] = field(default_factory=list)
    confs_by_step: list[float] = field(default_factory=list)
    unmask_step: int | None = None

    @property
    def num_pred_changes(self) -> int:
        if not self.preds_by_step:
            return 0
        first = self.preds_by_step[0]
        return sum(1 for p in self.preds_by_step[1:] if p != first)

    @property
    def changed_vs_first(self) -> bool:
        return self.num_pred_changes > 0

    @property
    def committed_differs_from_first(self) -> bool:
        if self.first_pred is None or self.final_pred is None:
            return False
        return self.final_pred != self.first_pred


@dataclass
class BlockTrace:
    block_idx: int
    gen_start_abs: int
    gen_end_abs: int
    inner_steps: int = 0
    positions: list[PositionTrace] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        n = len(self.positions)
        if n == 0:
            return {
                "block_idx": self.block_idx,
                "n_positions": 0,
                "inner_steps": self.inner_steps,
                "frac_changed_vs_first": 0.0,
                "frac_committed_differs": 0.0,
                "mean_pred_changes": 0.0,
            }
        changed = sum(p.changed_vs_first for p in self.positions)
        committed_diff = sum(p.committed_differs_from_first for p in self.positions)
        return {
            "block_idx": self.block_idx,
            "n_positions": n,
            "inner_steps": self.inner_steps,
            "frac_changed_vs_first": changed / n,
            "frac_committed_differs": committed_diff / n,
            "mean_pred_changes": sum(p.num_pred_changes for p in self.positions) / n,
        }


@dataclass
class SampleTrace:
    prompt_len: int
    blocks: list[BlockTrace] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)

    def aggregate_by_block(self) -> list[dict[str, Any]]:
        return [b.summary() for b in self.blocks]


@torch.no_grad()
def mdm_sample_traced(
    model,
    input_ids: torch.Tensor,
    *,
    tokenizer,
    block_size: int = 32,
    max_new_tokens: int = 2048,
    mask_id: int = FAST_DLLM_MASK_ID,
    stop_token: int = FAST_DLLM_STOP_TOKEN,
    small_block_size: int = 8,
    threshold: float = 1.0,
    top_p: float = 0.95,
    temperature: float = 0.0,
    use_block_cache: bool = False,
) -> tuple[torch.Tensor, SampleTrace]:
    """Single-sequence generation with per-block prediction volatility traces."""
    assert input_ids.shape[0] == 1, "tracing supports batch_size=1 only"
    device = input_ids.device
    seq_len = torch.tensor([input_ids.shape[1]], device=device)
    min_len = input_ids.shape[1]

    num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
    num_small_blocks = block_size // small_block_size
    prompt_len = input_ids.shape[1]
    trace = SampleTrace(prompt_len=prompt_len)

    # Warm-start partial block (same as upstream batch_sample)
    if min_len > block_size:
        output = model.forward(
            input_ids=input_ids[:, : (min_len // block_size * block_size)],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )
        logits, past_key_values = output.logits, output.past_key_values
        if min_len % block_size == 0:
            next_token = logits[:, -1:, :].argmax(dim=-1)
            if input_ids.shape[1] <= min_len:
                input_ids = torch.cat([input_ids, next_token], dim=1)
            else:
                input_ids[0, min_len] = next_token.squeeze(dim=-1)
    else:
        past_key_values = None

    seq_block_idx = seq_len // block_size
    start_block_idx = min_len // block_size
    gen_block_counter = 0

    for block_idx in range(start_block_idx, num_blocks):
        if stop_token in input_ids[:, prompt_len:]:
            break

        block_trace = BlockTrace(
            block_idx=gen_block_counter,
            gen_start_abs=max(prompt_len, block_idx * block_size),
            gen_end_abs=(block_idx + 1) * block_size,
        )
        pos_traces: dict[int, PositionTrace] = {}
        inner_step = 0

        if (seq_block_idx == block_idx).all():
            pad = mask_id * torch.ones(
                (1, block_size - input_ids.shape[1] % block_size),
                device=device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_ids, pad], dim=1)
            input_ids = x_init
        else:
            x_init = input_ids[:, : (block_idx + 1) * block_size]

        x_t = x_init.clone()
        block_past_key_values = None

        while True:
            mask_idx = x_t[:, -block_size:] == mask_id
            if mask_idx.sum() == 0:
                output = model.forward(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits, past_key_values = output.logits, output.past_key_values
                next_token = logits[:, -1:, :].argmax(dim=-1)
                x_t = torch.cat([x_t, next_token], dim=1)
                inner_step += 1
                break

            for small_block_idx in range(num_small_blocks):
                small_block_start_idx = small_block_idx * small_block_size
                small_block_end_idx = small_block_start_idx + small_block_size
                start = -block_size + small_block_start_idx
                end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                while True:
                    mask_idx = x_t[:, -block_size:] == mask_id
                    if mask_idx[:, start:end].sum() == 0:
                        break

                    if use_block_cache:
                        if block_past_key_values is None or (
                            x_t[:, -block_size + small_block_start_idx] == mask_id
                        ).any():
                            output = model.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                            )
                            logits, block_past_key_values = (
                                output.logits,
                                output.block_past_key_values,
                            )
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        else:
                            logits = model.forward(
                                input_ids=x_t[:, start:end],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_past_key_values=block_past_key_values,
                                replace_position=small_block_start_idx,
                            ).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                    else:
                        logits = model.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                        ).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                    x_1, p_1t = model.sample_with_top_p(
                        logits, top_p=top_p, temperature=temperature
                    )
                    x1_p = torch.squeeze(
                        torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1
                    )
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                    # Record argmax predictions for currently masked slots in this slice
                    block_abs_start = x_t.shape[1] - block_size
                    slice_mask = mask_idx[:, start:end]
                    for col in range(x_1.shape[1]):
                        if not bool(slice_mask[0, col].item()):
                            continue
                        abs_pos = block_abs_start + small_block_start_idx + col
                        if abs_pos < prompt_len:
                            continue
                        pred = int(x_1[0, col].item())

                        if abs_pos not in pos_traces:
                            pos_traces[abs_pos] = PositionTrace(
                                abs_pos=abs_pos,
                                rel_gen_pos=abs_pos - prompt_len,
                                pos_in_block=small_block_start_idx + col,
                            )
                        pt = pos_traces[abs_pos]
                        if pt.first_pred is None:
                            pt.first_pred = pred
                        pt.preds_by_step.append(pred)
                        pt.confs_by_step.append(float(x1_p[0, col].item()))

                    unmask_idx = x1_p > threshold
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]

                    for col in range(x_1.shape[1]):
                        if not bool(unmask_idx[0, col].item()):
                            continue
                        abs_pos = block_abs_start + small_block_start_idx + col
                        if abs_pos in pos_traces and pos_traces[abs_pos].unmask_step is None:
                            pos_traces[abs_pos].unmask_step = inner_step
                            pos_traces[abs_pos].final_pred = int(x_1[0, col].item())

                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                    inner_step += 1

                    if stop_token in x_t[:, prompt_len:]:
                        stop_rel = (x_t[:, prompt_len:] == stop_token).nonzero()
                        if stop_rel.numel() > 0:
                            stop_rel_idx = int(stop_rel[0, 1].item())
                            tail = x_t[:, prompt_len : prompt_len + stop_rel_idx]
                            if (tail == mask_id).sum() == 0:
                                break

        block_trace.inner_steps = inner_step
        block_trace.positions = [
            pos_traces[k] for k in sorted(pos_traces.keys())
        ]
        if block_trace.positions:
            trace.blocks.append(block_trace)
            gen_block_counter += 1

        if input_ids.shape[1] == x_t.shape[1]:
            input_ids = x_t
        else:
            input_ids[:, : (block_idx + 1) * block_size] = x_t[:, :-1]
            if (seq_block_idx == block_idx).all():
                input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
            else:
                if input_ids.shape[1] <= (block_idx + 1) * block_size:
                    input_ids = x_t
                else:
                    sel = seq_block_idx == block_idx
                    input_ids[sel, (block_idx + 1) * block_size] = x_t[
                        sel, (block_idx + 1) * block_size
                    ]
        seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

        if stop_token in input_ids[:, prompt_len:]:
            break

    if stop_token in input_ids[:, prompt_len:]:
        stop_rel = (input_ids[:, prompt_len:] == stop_token).nonzero()
        if stop_rel.numel() > 0:
            stop_rel_idx = int(stop_rel[0, 1].item())
            input_ids = input_ids[:, : prompt_len + stop_rel_idx + 1]

    trace.output_ids = input_ids[0, prompt_len:].tolist()
    return input_ids, trace


def trace_to_jsonable(trace: SampleTrace, tokenizer) -> dict[str, Any]:
    blocks = []
    for bt in trace.blocks:
        positions = []
        for p in bt.positions:
            positions.append(
                {
                    "abs_pos": p.abs_pos,
                    "rel_gen_pos": p.rel_gen_pos,
                    "pos_in_block": p.pos_in_block,
                    "first_pred_id": p.first_pred,
                    "first_pred": tokenizer.decode([p.first_pred]) if p.first_pred is not None else None,
                    "final_pred_id": p.final_pred,
                    "final_pred": tokenizer.decode([p.final_pred]) if p.final_pred is not None else None,
                    "num_pred_changes": p.num_pred_changes,
                    "changed_vs_first": p.changed_vs_first,
                    "committed_differs_from_first": p.committed_differs_from_first,
                    "unmask_step": p.unmask_step,
                    "n_forward_passes": len(p.preds_by_step),
                    "preds_by_step": p.preds_by_step,
                    "conf_by_step": p.confs_by_step,
                }
            )
        blocks.append(
            {
                **bt.summary(),
                "positions": positions,
            }
        )
    return {
        "prompt_len": trace.prompt_len,
        "n_gen_tokens": len(trace.output_ids),
        "blocks": blocks,
    }
