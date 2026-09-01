"""Sparse-KV variant of Dream's block_diffusion_generate.

Forked from Dream-org/DreamReasoner-8B/generation_utils.py. Adds a two-pass
per-block scheme compatible with sparse_kv_exp's selector/attention_hook stack:

- prefill: standard, populates DynamicCache with prompt K/V.
- for each gen block:
    step 0 (selector): run one full-cache forward with capture_attention active.
                       score prompt-range attention (from middle query) and pick
                       top-K keys per layer.
    steps 1..N (exec): run denoising forwards with attention_experiment(sparse=True)
                       so old (prompt) keys outside keep-set are masked out.

Only the `fp16_middle` selector variant is supported here — no KV-store
requantization, no quant K/V. This is a pilot to see if sparse transfers to
Dream at all.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

_SRC_DIR = Path(__file__).resolve().parent



if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fast_dllm_attn_capture import capture_attention  # noqa: E402

from attention_hook import attention_experiment, configure_from_model  # noqa: E402
from kv_store import DualPrecisionCache  # noqa: E402

# --- Copied unchanged from Dream's generation_utils.py ----------------------


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        token = torch.multinomial(probs, num_samples=1)
    else:
        token = probs.argmax(dim=-1, keepdim=True)
    token_prob = torch.gather(probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def build_block_diffusion_attention_mask(
    num_blocks: int, block_length: int, device: torch.device, batch_size: int = 1
) -> torch.Tensor:
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    return (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )


def _resolve_stopping_ids(x):
    if x is None:
        return None
    if isinstance(x, int):
        return [x]
    return list(x)


def _should_stop(generated_ids, prompt_length, stopping_criteria_idx):
    if not stopping_criteria_idx:
        return False
    gen_part = generated_ids[:, prompt_length:]
    return any((gen_part == s).any().item() for s in stopping_criteria_idx)


def _select_transfer_index(
    remasking_strategy,
    mask_index,
    x0,
    x0_p,
    num_transfer_tokens,
    step,
    confidence_threshold,
    eb_threshold,
    *,
    force_accept=False,
):
    if force_accept:
        return mask_index.clone()
    if remasking_strategy == "low_confidence_dynamic":
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        k = max(1, int(num_transfer_tokens[step].item()))
        for j in range(confidence.shape[0]):
            high_conf_mask = confidence[j] > confidence_threshold
            if int(high_conf_mask.sum().item()) >= k:
                transfer_index[j] = high_conf_mask
            else:
                _, idx = torch.topk(confidence[j], k)
                transfer_index[j, idx] = True
        return transfer_index
    raise ValueError(f"only low_confidence_dynamic supported, got {remasking_strategy!r}")


# --- Sparse-KV additions ----------------------------------------------------


def _pick_topk_keep(
    captured: dict[int, torch.Tensor],
    prompt_len: int,
    topk: int,
    mode: str = "middle",
) -> dict[int, list[int]]:
    """From captured attention pick top-K prompt keys.

    mode="middle" — attention row of the middle query token (selector.py mode='middle').
    mode="all_mean" — mean attention across all queries in the block (mode='all_mean').
    """
    keep: dict[int, list[int]] = {}
    for lid, w in captured.items():
        if w.dim() == 4:  # [B, H, T_q, T_k]
            w = w.mean(dim=1)
        b, t_q, t_k = w.shape
        if mode == "middle":
            mid = t_q // 2
            row = w[0, mid, :prompt_len]
        elif mode == "all_mean":
            row = w[0, :, :prompt_len].mean(dim=0)
        else:
            raise ValueError(f"unknown selector mode {mode!r}")
        k = min(topk, prompt_len)
        _, idx = torch.topk(row, k)
        keep[lid] = sorted(idx.tolist())
    return keep


@torch.no_grad()
def block_diffusion_generate_sparse(
    model,
    input_ids: torch.LongTensor,
    mask_id: int,
    gen_length: int = 128,
    block_length: Optional[int] = None,
    denoising_steps: Optional[int] = None,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.9,
    eb_threshold: Optional[float] = 0.35,
    stopping_criteria_idx: Optional[Union[int, Sequence[int]]] = None,
    return_dict_in_generate: bool = False,
    # sparse-specific
    sparse_topk: int = 64,
    selector_mode: str = "middle",
    selector_k_bits: int = 16,
    selector_v_bits: int = 16,
    exec_k_bits: int = 16,
    exec_v_bits: int = 16,
    kivi_group_size: int = 32,
    kivi_residual_length: int = 32,
    quant_scheme: str = "kivi",
) -> tuple[torch.LongTensor, int]:
    """Sparse-KV block diffusion. Returns (sequences, nfe).

    Any of selector_k/v_bits, exec_k/v_bits < 16 activates DualPrecisionCache
    with KIVI-quantized views (dequantized back to fp16 with quant noise). The
    k4sel_v4 config corresponds to all four = 4.
    """
    if selector_mode not in {"middle", "all_mean"}:
        raise NotImplementedError(f"selector_mode={selector_mode!r} unsupported")
    use_kv_store = min(selector_k_bits, selector_v_bits, exec_k_bits, exec_v_bits) < 16
    model.eval()
    device = input_ids.device
    batch_size, prompt_length = input_ids.shape
    if batch_size != 1:
        raise ValueError("sparse pilot supports batch_size=1")
    block_length = block_length or getattr(model.config, "block_size", 32)
    if denoising_steps is None:
        denoising_steps = block_length
    stopping_criteria_idx = _resolve_stopping_ids(stopping_criteria_idx)
    configure_from_model(model)

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    attn_mask_full = build_block_diffusion_attention_mask(
        num_blocks, block_length, device, batch_size=batch_size
    )
    position_ids = (
        torch.arange(total_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )

    x = torch.full((batch_size, total_length), mask_id, dtype=input_ids.dtype, device=device)
    x[:, :prompt_length] = input_ids

    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    past_key_values = DynamicCache()
    kv_store = DualPrecisionCache() if use_kv_store else None
    nfe = 0

    # Prefill (dense — no sparse yet since cache_len=0)
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        model(
            cur_x,
            attention_mask=attn_mask_full[:, :prefill_length, :prefill_length],
            position_ids=position_ids[:, :prefill_length],
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )
        nfe += 1

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = block_start + block_length
        cur_x = x[:, block_start:block_end].clone()
        # "Old" cache = everything before current block: prompt-prefix + all
        # previously-generated blocks. Matches Fast-dLLM's sparse_old_cache=True
        # semantics (cache_len grows with each block, keep-set is re-picked per
        # block using the middle-query attention row over the full old cache).
        old_cache_len = past_key_values.get_seq_length() if len(past_key_values) > 0 else 0

        cur_attn_mask = attn_mask_full[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]

        # --- Build KIVI precision views BEFORE selector-forward so that
        # selector attention uses quantized K/V if configured (k4sel_v4).
        if use_kv_store and old_cache_len > 0:
            kv_store.bind_fp16(past_key_values)
            kv_store.build_precision_views(
                selector_k_bits=selector_k_bits,
                selector_v_bits=selector_v_bits,
                exec_k_bits=exec_k_bits,
                exec_v_bits=exec_v_bits,
                kivi_group_size=kivi_group_size,
                kivi_residual_length=kivi_residual_length,
                keys_pre_rope=False,
                k_quant_scheme=quant_scheme,
                v_quant_scheme=quant_scheme,
                model=model,
            )

        # --- Selector step: forward + capture. K/V precision comes from
        # kv_store.selector_k/v views (via attention_experiment ctx below).
        per_layer_keep: dict[int, list[int]] = {}
        if old_cache_len > 0 and sparse_topk < old_cache_len:
            sel_ctx = (
                attention_experiment(
                    cache_len=old_cache_len,
                    current_block_len=block_length,
                    num_queries=block_length,
                    phase="selector",
                    sparse=False,
                    per_layer_keep=None,
                    per_head_sparse=False,
                    q_bits=16,
                    k_bits=selector_k_bits,
                    v_bits=selector_v_bits,
                    kv_store=kv_store,
                    cost_summary=None,
                    enabled=True,
                )
                if use_kv_store
                else nullcontext()
            )
            # ORDER MATTERS: capture must be OUTER so that fast_dllm captures
            # attention weights AFTER attention_hook substitutes quantized K/V
            # from kv_store — otherwise keep-set is picked from fp16 attention
            # even when selector_k_bits=4.
            with capture_attention(None, head_mean=True) as captured, sel_ctx:
                logits = model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                ).logits
                # capture_attention's finally-block clears _CAPTURED, so snapshot inside
                snap = {k: v.clone() for k, v in captured.items()}
            per_layer_keep = _pick_topk_keep(
                snap,
                prompt_len=old_cache_len,
                topk=sparse_topk,
                mode=selector_mode,
            )
        else:
            logits = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            ).logits
        nfe += 1

        # First denoising step uses selector-forward logits.
        mask_index = cur_x == mask_id
        x0, x0_p = sample_with_temperature_topk_topp(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        x0 = torch.where(mask_index, x0, cur_x)
        transfer_index = _select_transfer_index(
            remasking_strategy,
            mask_index,
            x0,
            x0_p,
            num_transfer_tokens,
            step=0,
            confidence_threshold=confidence_threshold,
            eb_threshold=eb_threshold,
            force_accept=(denoising_steps == 1),
        )
        cur_x[transfer_index] = x0[transfer_index]

        # Build sparse exec override (per-layer indexed) after selector-set is chosen.
        # Precision views already built above (before selector-forward).
        if use_kv_store and old_cache_len > 0 and per_layer_keep:
            kv_store.build_sparse_exec(
                per_layer_keep,
                requantize=False,
                source="exec",
                k_bits=exec_k_bits,
                v_bits=exec_v_bits,
                kivi_group_size=kivi_group_size,
                kivi_residual_length=kivi_residual_length,
                k_quant_scheme=quant_scheme,
                v_quant_scheme=quant_scheme,
            )

        # --- Exec steps with sparse attention on the full old cache
        # (prompt-prefix + all prior gen blocks).
        for step in range(1, denoising_steps + 1):
            mask_index = cur_x == mask_id
            done = int(mask_index.sum().item()) == 0
            force_accept = step == denoising_steps - 1

            # sparse experiment context — only masks columns 0..old_cache_len
            ctx = (
                attention_experiment(
                    cache_len=old_cache_len,
                    current_block_len=block_length,
                    num_queries=block_length,
                    phase="exec",
                    sparse=bool(per_layer_keep),
                    per_layer_keep=per_layer_keep if per_layer_keep else None,
                    per_head_sparse=False,
                    q_bits=16,
                    k_bits=exec_k_bits,
                    v_bits=exec_v_bits,
                    kv_store=kv_store if use_kv_store else None,
                    cost_summary=None,
                    enabled=bool(per_layer_keep),
                )
                if per_layer_keep
                else nullcontext()
            )
            with ctx:
                if done:
                    # finalize KV for this block
                    model(
                        cur_x,
                        attention_mask=cur_attn_mask,
                        position_ids=cur_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True,
                    )
                    nfe += 1
                    break
                logits = model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=(step == denoising_steps),
                ).logits
            nfe += 1
            x0, x0_p = sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            x0 = torch.where(mask_index, x0, cur_x)
            transfer_index = _select_transfer_index(
                remasking_strategy,
                mask_index,
                x0,
                x0_p,
                num_transfer_tokens,
                step=min(step, denoising_steps - 1),
                confidence_threshold=confidence_threshold,
                eb_threshold=eb_threshold,
                force_accept=force_accept,
            )
            cur_x[transfer_index] = x0[transfer_index]

        x[:, block_start:block_end] = cur_x

        if _should_stop(x, prompt_length, stopping_criteria_idx):
            break

    output_length = min(total_length, prompt_length + gen_length)
    return x[:, :output_length], nfe
