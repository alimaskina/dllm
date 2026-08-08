"""
Block-diffusion generation with cross-block attention capture on first denoising step.

For each generation block B_j, on inner_step==0 capture attention from B_j queries
to all prior generation blocks B_i (i < j).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from fast_dllm_attn_capture import aggregate_layers, block_saliency, capture_attention
from generation_traced import (
    FAST_DLLM_MASK_ID,
    FAST_DLLM_STOP_TOKEN,
    BlockTrace,
    PositionTrace,
    SampleTrace,
    mdm_sample_traced,
    trace_to_jsonable,
)

__all__ = [
    "CrossBlockAttnRecord",
    "SampleAttnTrace",
    "mdm_sample_attn_traced",
    "attn_trace_to_jsonable",
    "trace_to_jsonable",
]


@dataclass
class CrossBlockAttnRecord:
    source_block: int
    observer_block: int
    distance: int
    source_abs_start: int
    source_abs_end: int
    saliency: list[float]


@dataclass
class SampleAttnTrace:
    prompt_len: int
    block_size: int
    records: list[CrossBlockAttnRecord] = field(default_factory=list)
    generation_trace: SampleTrace | None = None

    def records_for_source(self, source_block: int) -> list[CrossBlockAttnRecord]:
        return [r for r in self.records if r.source_block == source_block]


def _capture_cross_block_attn(
    captured: dict[int, torch.Tensor],
    *,
    observer_block: int,
    prior_blocks: list[tuple[int, int]],
    block_size: int,
) -> list[CrossBlockAttnRecord]:
    attn = aggregate_layers(captured).cpu()
    # With KV cache: [T_q=block_size, T_k=past_len+block_size]; queries are local 0..bd-1.
    query_start = 0
    query_end = block_size
    records: list[CrossBlockAttnRecord] = []

    for source_block, (source_abs_start, source_abs_end) in enumerate(prior_blocks):
        sal = block_saliency(
            attn,
            query_start=query_start,
            query_end=query_end,
            key_start=source_abs_start,
            key_end=source_abs_end,
        )
        records.append(
            CrossBlockAttnRecord(
                source_block=source_block,
                observer_block=observer_block,
                distance=observer_block - source_block,
                source_abs_start=source_abs_start,
                source_abs_end=source_abs_end,
                saliency=sal,
            )
        )
    return records


@torch.no_grad()
def mdm_sample_attn_traced(
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
    attn_layers: set[int] | None = None,
) -> tuple[torch.Tensor, SampleAttnTrace]:
    """Generation with cross-block attention on first denoising step per block."""
    assert input_ids.shape[0] == 1
    device = input_ids.device
    seq_len = torch.tensor([input_ids.shape[1]], device=device)
    min_len = input_ids.shape[1]

    num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
    num_small_blocks = block_size // small_block_size
    prompt_len = input_ids.shape[1]
    gen_trace = SampleTrace(prompt_len=prompt_len)
    attn_trace = SampleAttnTrace(prompt_len=prompt_len, block_size=block_size)

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
    completed_blocks: list[tuple[int, int]] = []

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
        captured_first_step = False

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

                    capture_now = (
                        inner_step == 0
                        and not captured_first_step
                        and gen_block_counter > 0
                    )

                    if use_block_cache:
                        if block_past_key_values is None or (
                            x_t[:, -block_size + small_block_start_idx] == mask_id
                        ).any():
                            fwd_kwargs = dict(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                            )
                            if capture_now:
                                with capture_attention(attn_layers) as captured:
                                    output = model.forward(**fwd_kwargs, block_size=block_size)
                                    if captured:
                                        attn_trace.records.extend(
                                            _capture_cross_block_attn(
                                                captured,
                                                observer_block=gen_block_counter,
                                                prior_blocks=completed_blocks,
                                                block_size=block_size,
                                            )
                                        )
                                        captured_first_step = True
                            else:
                                output = model.forward(**fwd_kwargs, block_size=block_size)
                            logits, block_past_key_values = (
                                output.logits,
                                output.block_past_key_values,
                            )
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        else:
                            fwd_kwargs = dict(
                                input_ids=x_t[:, start:end],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_past_key_values=block_past_key_values,
                                replace_position=small_block_start_idx,
                            )
                            if capture_now:
                                with capture_attention(attn_layers) as captured:
                                    output = model.forward(**fwd_kwargs, block_size=block_size)
                                    if captured:
                                        attn_trace.records.extend(
                                            _capture_cross_block_attn(
                                                captured,
                                                observer_block=gen_block_counter,
                                                prior_blocks=completed_blocks,
                                                block_size=block_size,
                                            )
                                        )
                                        captured_first_step = True
                                logits = output.logits
                            else:
                                logits = model.forward(**fwd_kwargs, block_size=block_size).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                    else:
                        fwd_kwargs = dict(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                        )
                        if capture_now:
                            with capture_attention(attn_layers) as captured:
                                output = model.forward(**fwd_kwargs, block_size=block_size)
                                if captured:
                                    attn_trace.records.extend(
                                        _capture_cross_block_attn(
                                            captured,
                                            observer_block=gen_block_counter,
                                            prior_blocks=completed_blocks,
                                            block_size=block_size,
                                        )
                                    )
                                    captured_first_step = True
                            logits = output.logits
                        else:
                            logits = model.forward(**fwd_kwargs, block_size=block_size).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                    x_1, p_1t = model.sample_with_top_p(
                        logits, top_p=top_p, temperature=temperature
                    )
                    x1_p = torch.squeeze(
                        torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1
                    )
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

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
        block_trace.positions = [pos_traces[k] for k in sorted(pos_traces.keys())]
        if block_trace.positions:
            gen_trace.blocks.append(block_trace)
            completed_blocks.append((block_trace.gen_start_abs, block_trace.gen_end_abs))
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

    gen_trace.output_ids = input_ids[0, prompt_len:].tolist()
    attn_trace.generation_trace = gen_trace
    return input_ids, attn_trace


def attn_trace_to_jsonable(attn_trace: SampleAttnTrace) -> dict[str, Any]:
    return {
        "prompt_len": attn_trace.prompt_len,
        "block_size": attn_trace.block_size,
        "records": [
            {
                "source_block": r.source_block,
                "observer_block": r.observer_block,
                "distance": r.distance,
                "source_abs_start": r.source_abs_start,
                "source_abs_end": r.source_abs_end,
                "saliency": r.saliency,
            }
            for r in attn_trace.records
        ],
    }
