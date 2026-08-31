"""Quick verification: does Dream sparse actually mask growing cache?"""
import sys, os
sys.path.insert(0, '/home/alimaskina/dllm/block_diffusion/sparse_kv_exp')
sys.path.insert(0, '/home/alimaskina/dllm/block_diffusion')
os.chdir('/home/alimaskina/dllm/block_diffusion/sparse_kv_exp')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Monkey-patch dream_sparse_generate to print per-block stats
import dream_sparse_generate as dsg

_orig = dsg.block_diffusion_generate_sparse

@torch.no_grad()
def patched(model, input_ids, mask_id, gen_length, block_length, sparse_topk, selector_mode, **kw):
    from transformers.cache_utils import DynamicCache
    from attention_hook import attention_experiment, configure_from_model
    from fast_dllm_attn_capture import capture_attention
    from contextlib import nullcontext

    model.eval()
    device = input_ids.device
    batch_size, prompt_length = input_ids.shape
    denoising_steps = block_length
    configure_from_model(model)

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    attn_mask_full = dsg.build_block_diffusion_attention_mask(num_blocks, block_length, device, batch_size=batch_size)
    position_ids = torch.arange(total_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    x = torch.full((batch_size, total_length), mask_id, dtype=input_ids.dtype, device=device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    past_key_values = DynamicCache()

    print(f"[verify] prompt_len={prompt_length}  prefill_blocks={prefill_blocks}  prefill_length={prefill_length}  gen_length={gen_length}  block_length={block_length}  k={sparse_topk}")

    if prefill_length > 0:
        model(x[:, :prefill_length], attention_mask=attn_mask_full[:, :prefill_length, :prefill_length],
              position_ids=position_ids[:, :prefill_length], past_key_values=past_key_values,
              use_cache=True, store_kv=True)

    num_transfer_tokens = dsg.get_num_transfer_tokens(block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = block_start + block_length
        cur_x = x[:, block_start:block_end].clone()
        old_cache_len = past_key_values.get_seq_length() if len(past_key_values) > 0 else 0
        cur_attn_mask = attn_mask_full[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]

        per_layer_keep = {}
        sparse_will_fire = old_cache_len > 0 and sparse_topk < old_cache_len

        if sparse_will_fire:
            with capture_attention(None, head_mean=True) as captured:
                logits = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                               past_key_values=past_key_values, use_cache=True, store_kv=False).logits
                snap = {k: v.clone() for k, v in captured.items()}
            per_layer_keep = dsg._pick_middle_topk_keep(snap, prompt_len=old_cache_len, topk=sparse_topk)
            # Sanity: how many unique layers, what's the range of kept indices
            layers = sorted(per_layer_keep.keys())
            keep_l0 = per_layer_keep[layers[0]]
            keep_lmid = per_layer_keep[layers[len(layers)//2]]
            print(f"[block {num_block}] old_cache={old_cache_len:4d} SPARSE_FIRED k={sparse_topk} "
                  f"layers={len(layers)} keep_L0=[{min(keep_l0)},{max(keep_l0)}] mid=[{min(keep_lmid)},{max(keep_lmid)}] "
                  f"|kept_range/cache|={max(keep_lmid)-min(keep_lmid)}/{old_cache_len}")
        else:
            logits = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                           past_key_values=past_key_values, use_cache=True, store_kv=False).logits
            print(f"[block {num_block}] old_cache={old_cache_len:4d} sparse_skipped (k={sparse_topk} >= cache)")

        mask_index = cur_x == mask_id
        x0, x0_p = dsg.sample_with_temperature_topk_topp(logits, temperature=0.0, top_k=0, top_p=1.0)
        x0 = torch.where(mask_index, x0, cur_x)
        transfer_index = dsg._select_transfer_index("low_confidence_dynamic", mask_index, x0, x0_p,
                                                    num_transfer_tokens, step=0,
                                                    confidence_threshold=0.9, eb_threshold=0.35,
                                                    force_accept=(denoising_steps == 1))
        cur_x[transfer_index] = x0[transfer_index]

        for step in range(1, denoising_steps + 1):
            mask_index = cur_x == mask_id
            done = int(mask_index.sum().item()) == 0
            force_accept = step == denoising_steps - 1
            ctx = (attention_experiment(cache_len=old_cache_len, current_block_len=block_length,
                                        num_queries=block_length, phase="exec", sparse=bool(per_layer_keep),
                                        per_layer_keep=per_layer_keep if per_layer_keep else None,
                                        per_head_sparse=False, q_bits=16, k_bits=16, v_bits=16,
                                        kv_store=None, cost_summary=None, enabled=bool(per_layer_keep))
                   if per_layer_keep else nullcontext())
            with ctx:
                if done:
                    model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                          past_key_values=past_key_values, use_cache=True, store_kv=True)
                    break
                logits = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                               past_key_values=past_key_values, use_cache=True,
                               store_kv=(step == denoising_steps)).logits
            x0, x0_p = dsg.sample_with_temperature_topk_topp(logits, temperature=0.0, top_k=0, top_p=1.0)
            x0 = torch.where(mask_index, x0, cur_x)
            transfer_index = dsg._select_transfer_index("low_confidence_dynamic", mask_index, x0, x0_p,
                                                        num_transfer_tokens, step=min(step, denoising_steps-1),
                                                        confidence_threshold=0.9, eb_threshold=0.35,
                                                        force_accept=force_accept)
            cur_x[transfer_index] = x0[transfer_index]

        x[:, block_start:block_end] = cur_x

    return x[:, :min(total_length, prompt_length + gen_length)]


tok = AutoTokenizer.from_pretrained("Dream-org/DreamReasoner-8B", trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained("Dream-org/DreamReasoner-8B", trust_remote_code=True,
                                            torch_dtype=torch.bfloat16).eval().to("cuda:1")

q = "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?\nPlease reason step by step, and put your final answer within \\boxed{}."
prompt = tok.apply_chat_template([{"role":"user","content":q}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda:1")

print(f"\n### k=64 ###")
patched(mdl, ids, mask_id=mdl.config.mask_token_id, gen_length=512, block_length=32, sparse_topk=64, selector_mode="middle")

print(f"\n### k=32 ###")
patched(mdl, ids, mask_id=mdl.config.mask_token_id, gen_length=512, block_length=32, sparse_topk=32, selector_mode="middle")
