"""Block-diffusion generation with sparse old-cache + quant ablations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fast_dllm_attn_capture import capture_attention, captured_qk_snapshot  # noqa: E402

from attention_hook import attention_experiment, configure_from_model
from config import ExperimentConfig, parse_bits
from cost_model import RunCostSummary
from kv_store import DualPrecisionCache
from selector import select_per_layer, union_indices

FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645


def _build_run_state(exp_cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "config": exp_cfg.to_dict(),
        "blocks": [],
        "cost": RunCostSummary(),
        "kv_store": DualPrecisionCache(),
    }


@torch.no_grad()
def batch_sample_sparse_kv(
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
    *,
    exp_config: ExperimentConfig | None = None,
    experiment_log: list | None = None,
):
    """Fast-dLLM-v2 sampler with optional sparse old-cache + low-bit attention."""
    if input_ids.shape[0] != 1:
        raise ValueError("sparse_kv_exp requires batch_size=1")

    exp_cfg = exp_config or ExperimentConfig()
    state = _build_run_state(exp_cfg)
    configure_from_model(self)

    sel_prec = exp_cfg.selector_precision.as_ints()
    exec_prec = exp_cfg.exec_precision.as_ints()
    use_hook = exp_cfg.baseline != "original"
    hook_active = use_hook  # disabled per-forward when no old cache

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
        per_layer_selection: dict[int, Any] = {}
        per_layer_keep: dict[int, list[int]] = {}
        selection_done = False
        captured_snapshot: dict[int, torch.Tensor] | None = None
        qk_snapshot: dict[int, dict] | None = None

        kv_store: DualPrecisionCache = state["kv_store"]
        kv_store.bind_fp16(past_key_values)
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            kv_store.build_precision_views(
                selector_k_bits=sel_prec["k_bits"],
                selector_v_bits=sel_prec["v_bits"],
                exec_k_bits=exec_prec["k_bits"],
                exec_v_bits=exec_prec["v_bits"],
                k_group_size=1,
                keys_pre_rope=exp_cfg.keys_pre_rope,
                model=self,
            )

        block_log: dict[str, Any] = {
            "gen_block_idx": gen_block_idx,
            "denoising_steps": 0,
            "layers": {},
        }

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

                    cache_len = past_key_values.get_seq_length() if past_key_values else 0
                    is_selector_step = (
                        use_hook
                        and cache_len > 0
                        and step == 0
                        and not selection_done
                        and exp_cfg.sparse_old_cache
                    )
                    sparse_now = (
                        use_hook
                        and cache_len > 0
                        and selection_done
                        and exp_cfg.sparse_old_cache
                    )
                    phase = "selector" if is_selector_step else "exec"
                    prec = sel_prec if is_selector_step else exec_prec

                    capture_now = is_selector_step

                    ctx_attn = attention_experiment(
                        cache_len=cache_len,
                        current_block_len=block_size,
                        num_queries=block_size,
                        phase=phase,
                        sparse=sparse_now,
                        per_layer_keep=per_layer_keep if sparse_now else None,
                        per_head_sparse=exp_cfg.selector.per_head,
                        q_bits=prec["q_bits"],
                        k_bits=prec["k_bits"],
                        v_bits=prec["v_bits"],
                        kv_store=kv_store if use_hook else None,
                        cost_summary=state["cost"],
                        enabled=hook_active and cache_len > 0,
                    )

                    with ctx_attn:
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
                                    with capture_attention(
                                        None, head_mean=not exp_cfg.selector.per_head
                                    ) as captured:
                                        output = self.forward(**fwd_kwargs, block_size=block_size)
                                        captured_snapshot = {k: v.clone() for k, v in captured.items()}
                                        qk_snapshot = captured_qk_snapshot()
                                else:
                                    output = self.forward(**fwd_kwargs, block_size=block_size)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
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
                                    with capture_attention(
                                        None, head_mean=not exp_cfg.selector.per_head
                                    ) as captured:
                                        logits = self.forward(**fwd_kwargs, block_size=block_size).logits
                                        captured_snapshot = {k: v.clone() for k, v in captured.items()}
                                        qk_snapshot = captured_qk_snapshot()
                                else:
                                    logits = self.forward(**fwd_kwargs, block_size=block_size).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            fwd_kwargs = dict(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                            )
                            if capture_now:
                                with capture_attention(
                                    None, head_mean=not exp_cfg.selector.per_head
                                ) as captured:
                                    output = self.forward(**fwd_kwargs, block_size=block_size)
                                    captured_snapshot = {k: v.clone() for k, v in captured.items()}
                                    qk_snapshot = captured_qk_snapshot()
                                logits = output.logits
                            else:
                                logits = self.forward(**fwd_kwargs, block_size=block_size).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                    if capture_now and captured_snapshot:
                        num_cached = cache_len
                        masked_q = mask_idx[0].nonzero(as_tuple=True)[0].tolist()
                        exec_k_bits = exec_prec["k_bits"]
                        per_layer_selection = select_per_layer(
                            captured_snapshot,
                            num_cached=num_cached,
                            block_size=block_size,
                            selector=exp_cfg.selector,
                            masked_query_indices=masked_q,
                            captured_qk=qk_snapshot,
                            probe_old_k_bits=exec_k_bits if exec_k_bits < 16 else None,
                        )
                        per_layer_keep = {
                            lid: sel.selected_indices for lid, sel in per_layer_selection.items()
                        }
                        if exp_cfg.selector.per_head:
                            union = {
                                lid: union_indices(sel.selected_indices)  # type: ignore[arg-type]
                                for lid, sel in per_layer_selection.items()
                                if sel.per_head
                            }
                            kv_store.build_sparse_exec(union)
                        else:
                            kv_store.build_sparse_exec(per_layer_keep)  # type: ignore[arg-type]
                        selection_done = True
                        for lid, sel in per_layer_selection.items():
                            layer_rec = {
                                "old_cache_len": sel.old_cache_len,
                                "selected_k": sel.selected_k,
                                "topk_pct": exp_cfg.selector.topk_pct,
                                "per_head": sel.per_head,
                                "selector_mode": exp_cfg.selector.mode,
                                "selector_precision": sel_prec,
                                "execution_precision": exec_prec,
                                "attention_mass_captured": sel.mass_captured,
                                "selector_self_coverage": sel.selector_self_coverage,
                                "rank_overlap_at_k": sel.rank_overlap_at_k,
                                "quant_probe_coverage": sel.quant_probe_coverage,
                                "masked_query_count": len(masked_q),
                                "reference_queries": "full_masked_block",
                            }
                            if exp_cfg.log_selected_indices:
                                layer_rec["selected_indices"] = sel.selected_indices
                            block_log["layers"][str(lid)] = layer_rec

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
                    block_log["denoising_steps"] = step

        block_log["old_cache_len_at_start"] = (
            past_key_values.get_seq_length() if past_key_values else 0
        )
        state["blocks"].append(block_log)

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

            if past_key_values is not None:
                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

            finished_flag = finished_flag[~finished_flag]

    if len(finished_samples) < batch_size:
        for sample_idx in range(x_t.shape[0]):
            original_idx = sample_indices[sample_idx].item()
            finished_samples[original_idx] = x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)

    assert len(finished_samples) == batch_size
    if experiment_log is not None:
        state["cost_summary"] = state["cost"].aggregate(
            save_full_steps=exp_cfg.save_full_cost_steps
        )
        experiment_log.append(state)
    return finished_samples
