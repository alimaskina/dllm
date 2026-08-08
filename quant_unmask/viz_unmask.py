#!/usr/bin/env python3
"""
Visualize masked diffusion unmasking step by step.

Shows the completion sequence at each print_every steps:
  - newly revealed tokens: green
  - already unmasked:      white
  - still masked:          dim gray [?]

Usage:
  python viz_unmask.py --prompt "Janet's ducks lay 16 eggs per day."
  python viz_unmask.py --prompt "Solve: 2x + 3 = 11. x =" --comp-len 64 --n-steps 64 --print-every 8
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
import transformers.modeling_utils as _tmu

# LLaDA's custom class is incompatible with new transformers in two ways:
#   1. missing all_tied_weights_keys property
#   2. tie_weights() doesn't accept keyword args (missing_keys, recompute_mapping)
# Patch both methods once before any model load.

_orig_move = PreTrainedModel._move_missing_keys_from_meta_to_device
def _patched_move(self, *args, **kwargs):
    if not hasattr(self, "all_tied_weights_keys"):
        self.__class__.all_tied_weights_keys = property(lambda s: {})
    return _orig_move(self, *args, **kwargs)
PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move

_orig_finalize = PreTrainedModel._finalize_model_loading
@staticmethod
def _patched_finalize(model, load_config, loading_info):
    if not hasattr(model, "all_tied_weights_keys"):
        model.__class__.all_tied_weights_keys = property(lambda s: {})
    _orig_tie = model.__class__.tie_weights
    def _compat_tie(self, **kwargs):
        return _orig_tie(self)
    model.__class__.tie_weights = _compat_tie
    try:
        return _orig_finalize(model, load_config, loading_info)
    finally:
        model.__class__.tie_weights = _orig_tie
PreTrainedModel._finalize_model_loading = _patched_finalize

_orig_warmup = _tmu.caching_allocator_warmup
def _patched_warmup(model, *args, **kwargs):
    if not hasattr(model, "all_tied_weights_keys"):
        model.__class__.all_tied_weights_keys = property(lambda s: {})
    return _orig_warmup(model, *args, **kwargs)
_tmu.caching_allocator_warmup = _patched_warmup

# ANSI colours
GREEN  = "\033[92m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _num_transfer_tokens(mask_num: int, steps: int) -> list[int]:
    base = mask_num // steps
    remainder = mask_num % steps
    return [base + (1 if i < remainder else 0) for i in range(steps)]


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")


def render_state(step: int, token_strs: list[str], is_masked: list[bool],
                 newly_unmasked: set[int], comp_len: int):
    """Returns (terminal_line, html_row, md_block)."""
    term_parts, html_parts, md_parts = [], [], []
    for i in range(comp_len):
        if is_masked[i]:
            term_parts.append(f"{DIM}[?]{RESET}")
            html_parts.append('<span class="mask">[?]</span>')
            md_parts.append("▒")
        elif i in newly_unmasked:
            t = token_strs[i]
            term_parts.append(f"{GREEN}{t.replace(' ', '·')}{RESET}")
            html_parts.append(f'<span class="new">{_html_escape(t)}</span>')
            md_parts.append(f"**{t.strip()}**" if t.strip() else t)
        else:
            t = token_strs[i]
            term_parts.append(t)
            html_parts.append(f'<span class="done">{_html_escape(t)}</span>')
            md_parts.append(t)
    term_line = f"\n{BOLD}step {step:4d}{RESET}  " + "".join(term_parts)
    html_row = (f'<tr><td class="stepnum">step&nbsp;{step:4d}</td>'
                f'<td class="tokens">{"".join(html_parts)}</td></tr>')
    md_block = f"### step {step}\n\n" + "".join(md_parts) + "\n"
    return term_line, html_row, md_block


def print_state(step, token_strs, is_masked, newly_unmasked, comp_len):
    term_line, _, _ = render_state(step, token_strs, is_masked, newly_unmasked, comp_len)
    print(term_line)


@torch.no_grad()
def generate_viz(model, tok, prompt_ids: torch.Tensor, comp_len: int,
                 n_steps: int, mask_id: int, device: str,
                 print_every: int) -> tuple[torch.Tensor, list[str]]:
    """Returns (generated_ids, html_rows)."""
    comp = torch.full((comp_len,), mask_id, dtype=torch.long)
    ids = torch.cat([prompt_ids, comp]).unsqueeze(0).to(device)
    prompt_len = len(prompt_ids)

    transfer = _num_transfer_tokens(comp_len, n_steps)
    is_masked = [True] * comp_len
    token_strs = [""] * comp_len
    html_rows = []

    term_line, html_row, md_block = render_state(0, token_strs, is_masked, set(), comp_len)
    print(term_line)
    html_rows.append(html_row)
    md_blocks = [md_block]

    for step_idx in range(n_steps):
        out = model(ids)
        logits = (out.logits if hasattr(out, "logits") else out[0])[0]

        comp_logits = logits[prompt_len: prompt_len + comp_len].float()
        probs = F.softmax(comp_logits, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values

        masked_tensor = ids[0, prompt_len: prompt_len + comp_len] == mask_id
        confidence = torch.where(masked_tensor, conf,
                                 torch.full_like(conf, float("-inf")))

        k = min(transfer[step_idx], int(masked_tensor.sum()))
        newly = set()
        if k > 0:
            _, sel = torch.topk(confidence, k=k)
            ids[0, prompt_len + sel] = pred[sel]
            for idx in sel.tolist():
                is_masked[idx] = False
                token_strs[idx] = tok.decode([pred[idx].item()])
                newly.add(idx)

        step = step_idx + 1
        if step % print_every == 0 or step == n_steps:
            term_line, html_row, md_block = render_state(step, token_strs, is_masked, newly, comp_len)
            print(term_line)
            sys.stdout.flush()
            html_rows.append(html_row)
            md_blocks.append(md_block)

    return ids[0, prompt_len: prompt_len + comp_len], html_rows, md_blocks


_HTML_TEMPLATE = """\
<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Unmasking Visualization</title>
<style>
  body {{ font-family: monospace; font-size: 14px; background: #1e1e1e; color: #d4d4d4;
          padding: 20px; }}
  h2 {{ color: #9cdcfe; }}
  .meta {{ color: #888; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td {{ padding: 4px 8px; vertical-align: top; }}
  .stepnum {{ color: #888; white-space: nowrap; width: 80px; }}
  .tokens {{ word-break: break-all; line-height: 1.6; }}
  .mask {{ color: #555; }}
  .new {{ color: #4ec9b0; font-weight: bold; }}
  .done {{ color: #d4d4d4; }}
  hr {{ border-color: #444; }}
</style></head><body>
<h2>Unmasking: {model}</h2>
<div class="meta">
  <b>Prompt:</b> {prompt}<br>
  <b>comp_len</b>={comp_len} &nbsp; <b>n_steps</b>={n_steps} &nbsp;
  <b>quant</b>={quant}
</div>
<hr>
<table>{rows}</table>
<hr>
<div class="meta"><b>Final:</b> {final}</div>
</body></html>
"""


def _load_model(model_name, quant, device):
    from transformers import BitsAndBytesConfig
    kw = {"trust_remote_code": True}
    if quant in ("fp16", "bf16"):
        kw["torch_dtype"] = torch.float16 if quant == "fp16" else torch.bfloat16
        kw["low_cpu_mem_usage"] = False
        model = AutoModel.from_pretrained(model_name, **kw).to(device).eval()
        if not hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    elif quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw["device_map"] = "auto"
        model = AutoModel.from_pretrained(model_name, **kw).eval()
    elif quant == "int4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        kw["device_map"] = "auto"
        model = AutoModel.from_pretrained(model_name, **kw).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--prompt",      default=None)
    parser.add_argument("--gsm8k",       type=int, default=None,
                        help="Load N examples from GSM8K test set instead of --prompt")
    parser.add_argument("--gsm8k-seed",  type=int, default=42)
    parser.add_argument("--comp-len",    type=int, default=128)
    parser.add_argument("--n-steps",     type=int, default=128)
    parser.add_argument("--print-every", type=int, default=16)
    parser.add_argument("--mask-id",     type=int, default=126336)
    parser.add_argument("--quant",       default="fp16",
                        choices=["fp16", "bf16", "int8", "int4"])
    parser.add_argument("--out",         default=None,
                        help="Save to .md or .html")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build prompt list
    if args.gsm8k:
        from datasets import load_dataset
        ds_train = load_dataset("gsm8k", "main", split="train")
        ds_test  = load_dataset("gsm8k", "main", split="test")
        # 5-shot prefix (same examples as lm-eval-harness)
        prefix = "\n".join(
            f"Question: {ds_train[i]['question']}\nAnswer: {ds_train[i]['answer']}"
            for i in range(5)
        ) + "\n"
        rng = np.random.default_rng(args.gsm8k_seed)
        idx = rng.choice(len(ds_test), args.gsm8k, replace=False).tolist()
        examples = [
            (prefix + f"Question: {ds_test[i]['question']}\nAnswer:",
             ds_test[i]["answer"])
            for i in idx
        ]
    else:
        prompt = args.prompt or "Question: What is 2 + 2? Answer:"
        examples = [(prompt, None)]

    print(f"Loading model {args.model} ({args.quant})...")
    model = _load_model(args.model, args.quant, device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_md_sections = [
        f"# GSM8K Unmasking — {args.model}\n\n"
        f"**comp\\_len**={args.comp_len} · **n\\_steps**={args.n_steps} · "
        f"**quant**={args.quant} · **print\\_every**={args.print_every}\n\n"
        f"▒ = masked · **bold** = just revealed · plain = already open\n"
    ]

    for ex_idx, (question, gold) in enumerate(examples):
        prompt_ids = tok(question, return_tensors="pt",
                         add_special_tokens=False)["input_ids"][0]
        # show exact decoded prompt (what model actually receives)
        prompt_decoded = tok.decode(prompt_ids)

        print(f"\n{'═'*60}")
        print(f"Example {ex_idx+1}/{len(examples)}")
        print(f"Prompt ({len(prompt_ids)} tokens): {prompt_decoded}")
        if gold:
            print(f"Gold: {gold.split(chr(10))[-1]}")  # last line = #### N
        print(f"{DIM}[?] = masked   {GREEN}token{RESET} = just revealed   token = already open{RESET}")
        print("─" * 60)

        gen, html_rows, md_blocks = generate_viz(
            model, tok, prompt_ids, args.comp_len,
            args.n_steps, args.mask_id, device, args.print_every)

        final = tok.decode(gen, skip_special_tokens=True)
        print(f"\n{'─'*60}")
        print(f"Generated: {final}")
        if gold:
            print(f"Gold:      {gold}")

        section = (
            f"\n---\n\n## Example {ex_idx+1}\n\n"
            f"**Prompt** ({len(prompt_ids)} tokens):\n\n"
            f"```\n{prompt_decoded}\n```\n\n"
        )
        if gold:
            section += f"**Gold answer:** `{gold.split(chr(10))[-1].strip()}`\n\n"
        section += "---\n\n" + "\n---\n\n".join(md_blocks)
        section += f"\n---\n\n**Generated:** {final}\n"
        all_md_sections.append(section)

    if args.out and args.out.endswith(".md"):
        with open(args.out, "w") as f:
            f.write("\n".join(all_md_sections))
        print(f"\nSaved → {args.out}")
    elif args.out:
        html = _HTML_TEMPLATE.format(
            model=args.model,
            prompt=_html_escape(args.prompt),
            comp_len=args.comp_len,
            n_steps=args.n_steps,
            quant=args.quant,
            rows="\n".join(html_rows),
            final=_html_escape(final),
        )
        with open(args.out, "w") as f:
            f.write(html)
        print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
