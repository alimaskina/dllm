"""Block diffusion with step-0 top-K KV pruning for remaining denoising steps."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from kv_cache_quant import DEFAULT_KIVI_GROUP_SIZE, DEFAULT_KIVI_RESIDUAL_LENGTH
from topk_kv_policy import (
    aggregate_importance_across_layers,
    clone_dynamic_cache,
    keep_kv_indices,
    quantize_keep_tokens_kivi,
    select_top_k_indices,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fast_dllm_attn_capture import cached_token_importance, capture_attention  # noqa: E402

FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645


def _importance_from_capture(
    captured: dict[int, torch.Tensor],
    *,
    block_size: int,
) -> tuple[list[float], int]:
    if not captured:
        return [], 0
    sample_attn = next(iter(captured.values()))
    num_cached = int(sample_attn.shape[-1]) - block_size
    if num_cached <= 0:
        return [], 0
    per_layer = cached_token_importance(
        captured,
        num_cached=num_cached,
        query_start=0,
        query_end=block_size,
    )
    return aggregate_importance_across_layers(per_layer), num_cached


@torch.no_grad()
def batch_sample(
    self,
    input_ids,
    tokenizer,
    block_size,
    max_new_tokens,
    small_block_size,
    min_len,
    seq_len,
    mask_id=151665,
    threshold=0.95,
    stop_token=151645,
    use_block_cache=False,
    top_p=0.95,
    temperature=0.0,
    topk: int = 128,
    topk_bits: int = 0,
    keys_pre_rope: bool = False,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    prune_log: list | None = None,
):
    """Generation with per-block top-K cached-token attention after step-0 capture."""
    if input_ids.shape[0] != 1:
        raise ValueError("topk prune generation requires batch_size=1")

    num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
    batch_size = input_ids.shape[0]

    if min_len > block_size:
        output = self.forward(
            input_ids=input_ids[:, : (min_len // block_size * block_size)],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )
        logits, past_key_values = output.logits, output.past_key_values
        if min_len % block_size == 0:
            predict_sample_idx = seq_len == min_len
            predict_logits = logits[predict_sample_idx, -1:, :]
            next_token = predict_logits.argmax(dim=-1)
            if input_ids.shape[1] <= min_len:
                input_ids = torch.cat([input_ids, next_token], dim=1)
            else:
                input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
    else:
        past_key_values = None

    seq_block_idx = seq_len // block_size
    finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)
    start_block_idx = min_len // block_size
    num_small_blocks = block_size // small_block_size
    gen_block_idx = 0

    sample_indices = torch.arange(batch_size, device=self.device)
    finished_samples = {}

    for block_idx in range(start_block_idx, num_blocks):
        if finished_flag.all():
            break
        if (seq_block_idx == block_idx).all():
            x_init = mask_id * torch.ones(
                (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                device=self.device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_ids, x_init], dim=1)
            input_ids = x_init
        else:
            x_init = input_ids[:, : (block_idx + 1) * block_size]

        x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
        x_t = x_init.clone()
        step = 0
        block_past_key_values = None
        keep_indices: set[int] | None = None
        pruned_pkv = None

        while True:
            mask_idx = x_t[:, -block_size:] == mask_id
            if mask_idx.sum() == 0:
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                        stop_token_idx = (x_t[sample_idx, seq_len[sample_idx] :] == stop_token).nonzero()[0][0]
                        x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1 :] = tokenizer.pad_token_id
                if finished_flag.all():
                    break
                output = self.forward(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits, past_key_values = output.logits, output.past_key_values
                next_token = logits[:, -1:, :].argmax(dim=-1)
                next_token[finished_flag] = tokenizer.pad_token_id
                x_t = torch.cat([x_t, next_token], dim=1)
                step += 1
                if past_key_values is not None:
                    gen_block_idx += 1
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
                        past_key_values is not None
                        and step == 0
                        and keep_indices is None
                    )

                    pkv_fwd = pruned_pkv if (step > 0 and pruned_pkv is not None) else past_key_values
                    ctx = keep_kv_indices(self, keep_indices if step > 0 else None)

                    with ctx:
                        if use_block_cache:
                            if block_past_key_values is None or (
                                x_t[:, -block_size + small_block_start_idx] == mask_id
                            ).any():
                                fwd_kwargs = dict(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=pkv_fwd,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                )
                                if capture_now:
                                    with capture_attention(None) as captured:
                                        output = self.forward(**fwd_kwargs, block_size=block_size)
                                        captured_snapshot = {
                                            k: v.clone() for k, v in captured.items()
                                        }
                                else:
                                    output = self.forward(**fwd_kwargs, block_size=block_size)
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
                                    past_key_values=pkv_fwd,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                )
                                if capture_now:
                                    with capture_attention(None) as captured:
                                        logits = self.forward(**fwd_kwargs, block_size=block_size).logits
                                        captured_snapshot = {
                                            k: v.clone() for k, v in captured.items()
                                        }
                                else:
                                    logits = self.forward(**fwd_kwargs, block_size=block_size).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            fwd_kwargs = dict(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=pkv_fwd,
                                update_past_key_values=False,
                            )
                            if capture_now:
                                with capture_attention(None) as captured:
                                    output = self.forward(**fwd_kwargs, block_size=block_size)
                                    captured_snapshot = {
                                        k: v.clone() for k, v in captured.items()
                                    }
                                logits = output.logits
                            else:
                                logits = self.forward(**fwd_kwargs, block_size=block_size).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                    if capture_now:
                        importance, num_cached = _importance_from_capture(
                            captured_snapshot, block_size=block_size
                        )
                        keep_indices = select_top_k_indices(importance, topk)
                        pruned_pkv = clone_dynamic_cache(past_key_values)
                        quantize_keep_tokens_kivi(
                            pruned_pkv,
                            keep_indices,
                            topk_bits,
                            keys_pre_rope=keys_pre_rope,
                            model=self,
                            kivi_group_size=kivi_group_size,
                        )
                        if prune_log is not None:
                            prune_log.append(
                                {
                                    "block_idx": gen_block_idx,
                                    "num_cached": num_cached,
                                    "topk_requested": topk,
                                    "topk_kept": len(keep_indices),
                                    "topk_bits": topk_bits,
                                }
                            )

                    x_1, p_1t = self.sample_with_top_p(
                        logits, top_p=top_p, temperature=temperature
                    )
                    x1_p = torch.squeeze(
                        torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1
                    )
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                    unmask_idx = x1_p > threshold
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]

                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                    finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                    finished_flag = finished_flag | finished_row_flags

                    step += 1

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
                    input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[
                        seq_block_idx == block_idx, (block_idx + 1) * block_size
                    ]
        seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
        if finished_flag.any():
            for sample_idx in range(x_t.shape[0]):
                if finished_flag[sample_idx]:
                    original_idx = sample_indices[sample_idx].item()
                    finished_samples[original_idx] = x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)
            sample_indices = sample_indices[~finished_flag]
            input_ids = input_ids[~finished_flag]
            seq_block_idx = seq_block_idx[~finished_flag]
            seq_len = seq_len[~finished_flag]
            x_t = x_t[~finished_flag]

            for layer_id in range(len(past_key_values)):
                past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

            finished_flag = finished_flag[~finished_flag]

    if len(finished_samples) < batch_size:
        for sample_idx in range(x_t.shape[0]):
            original_idx = sample_indices[sample_idx].item()
            finished_samples[original_idx] = x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)

    assert len(finished_samples) == batch_size
    return finished_samples
