import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _trace_step(
    b,
    x,
    x0,
    confidence,
    transfer_index,
    num_transfer_tokens,
    block_i,
    global_step,
    num_block,
    prompt_len,
    gen_length,
    mask_id,
    pos_token_margin,
    pos_neg_entropy,
    top10,
    tokenizer,
):
    comp_start = prompt_len
    comp_slice = slice(comp_start, comp_start + gen_length)
    n_masked_before = int((x[b, comp_slice] == mask_id).sum().item())
    k = int(num_transfer_tokens[b, block_i].item())
    completion_tokens = [int(x[b, comp_start + rel].item()) for rel in range(gen_length)]
    unmasked = []
    for pos in transfer_index[b].nonzero(as_tuple=True)[0].tolist():
        tid = int(x0[b, pos].item())
        tok = tokenizer.decode([tid]) if tokenizer is not None else str(tid)
        unmasked.append({
            "pos": int(pos),
            "pos_comp": int(pos - comp_start),
            "token_id": tid,
            "token": tok,
            "confidence": float(confidence[b, pos].item()),
        })
    masked = []
    confidences = []
    predicted_token_id = []
    token_margins = []
    neg_entropies = []
    top10_token_id = []
    for rel_pos in range(gen_length):
        pos = comp_start + rel_pos
        is_masked = bool(x[b, pos].item() == mask_id)
        masked.append(is_masked)
        if is_masked:
            tid = int(x0[b, pos].item())
            confidences.append(float(confidence[b, pos].item()))
            predicted_token_id.append(tid)
            if pos_token_margin is not None:
                token_margins.append(float(pos_token_margin[b, pos].item()))
                neg_entropies.append(float(pos_neg_entropy[b, pos].item()))
                top10_token_id.append([int(t) for t in top10[b, pos].tolist()])
            else:
                token_margins.append(None)
                neg_entropies.append(None)
                top10_token_id.append(None)
        else:
            confidences.append(None)
            predicted_token_id.append(None)
            token_margins.append(None)
            neg_entropies.append(None)
            top10_token_id.append(None)
    gap = {
        "boundary_margin": None,
        "top1_pos_comp": None,
        "top2_pos_comp": None,
        "top1_confidence": None,
        "top2_confidence": None,
    }
    masked_conf = confidence[b, comp_slice].clone()
    masked_conf[x[b, comp_slice] != mask_id] = -np.inf
    n_masked = int((masked_conf > -np.inf).sum().item())
    if n_masked >= 2:
        vals, idxs = torch.topk(masked_conf, k=2)
        gap = {
            "boundary_margin": float((vals[0] - vals[1]).item()),
            "top1_pos_comp": int(idxs[0].item()),
            "top2_pos_comp": int(idxs[1].item()),
            "top1_confidence": float(vals[0].item()),
            "top2_confidence": float(vals[1].item()),
        }
    elif n_masked == 1:
        idx = int(masked_conf.argmax().item())
        gap = {
            "boundary_margin": None,
            "top1_pos_comp": idx,
            "top2_pos_comp": None,
            "top1_confidence": float(masked_conf[idx].item()),
            "top2_confidence": None,
        }
    return {
        "step": global_step,
        "block": int(num_block),
        "step_in_block": int(block_i),
        "k": k,
        "n_masked_before": n_masked_before,
        "mask_ratio": n_masked_before / gen_length,
        "unmasked": unmasked,
        "gap": gap,
        "completion_tokens": completion_tokens,
        "completion": {
            "masked": masked,
            "confidence": confidences,
            "predicted_token_id": predicted_token_id,
            "token_margin": token_margins,
            "neg_entropy": neg_entropies,
            "top10_token_id": top10_token_id,
        },
    }


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             record_trace=False, prompt_len=0, tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L) — left-padded if batched.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence', 'entropy', 'topk_margin', or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    trace = None
    global_steps = None
    if record_trace:
        trace = [[] for _ in range(prompt.shape[0])]
        global_steps = [0] * prompt.shape[0]
    if record_trace and prompt_len == 0:
        prompt_len = prompt.shape[1]

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'entropy':
                p = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                x0_p = (p * log_probs).sum(dim=-1)  # negative entropy; higher = more confident
            elif remasking == 'topk_margin':
                p = F.softmax(logits, dim=-1)
                top2 = p.topk(2, dim=-1).values
                x0_p = top2[:, :, 0] - top2[:, :, 1]
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            if record_trace:
                comp_start = prompt_len
                pos_token_margin = None
                pos_neg_entropy = None
                top10 = None
                if remasking in ('low_confidence', 'entropy', 'topk_margin'):
                    log_probs = F.log_softmax(logits, dim=-1)
                    top2 = p.topk(2, dim=-1)
                    pos_token_margin = top2.values[:, :, 0] - top2.values[:, :, 1]
                    pos_neg_entropy = (p * log_probs).sum(dim=-1)
                    top10 = p.topk(10, dim=-1).indices
                for b in range(x.shape[0]):
                    step_trace = _trace_step(
                        b,
                        x,
                        x0,
                        confidence,
                        transfer_index,
                        num_transfer_tokens,
                        i,
                        global_steps[b],
                        num_block,
                        prompt_len,
                        gen_length,
                        mask_id,
                        pos_token_margin,
                        pos_neg_entropy,
                        top10,
                        tokenizer,
                    )
                    trace[b].append(step_trace)
                    global_steps[b] += 1
            x[transfer_index] = x0[transfer_index]

    if record_trace:
        return x, trace
    return x
