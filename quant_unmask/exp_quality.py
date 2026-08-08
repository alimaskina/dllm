#!/usr/bin/env python3
"""
Quality evaluation: FP16 vs INT4 end-to-end generation on benchmark datasets.

Datasets:
  gsm8k      — math reasoning, eval: numeric answer accuracy (extract after ####)
  strategyqa — yes/no QA,      eval: yes/no accuracy
  sudoku     — 9×9 grid,       eval: valid solution rate

Generation: standard masked diffusion decoding (LLaDA / Dream style).
  - Start with all completion tokens masked
  - T steps: forward pass → unmask k = ceil(n_masked / remaining_steps) positions
    with highest confidence
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# ── dataset loaders (question, gold_answer) ─────────────────────────────────

def load_gsm8k(n, seed):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
    return [(ds[i]["question"], ds[i]["answer"]) for i in idx]


def load_strategyqa(n, seed):
    from datasets import load_dataset
    ds = load_dataset("metaeval/strategyqa", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
    rows = []
    for i in idx:
        ans = "Yes" if ds[i]["answer"] else "No"
        facts = ds[i].get("facts", [])
        gold = ans + ". " + " ".join(facts) if facts else ans
        rows.append((ds[i]["question"], gold))
    return rows


def load_sudoku(n, seed):
    from datasets import load_dataset
    ds = load_dataset("Ritvik19/sudoku-dataset", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
    def fmt(s81):
        return "\n".join(" ".join(s81[r*9:(r+1)*9]) for r in range(9))
    rows = []
    for i in idx:
        puzzle   = str(ds[i]["puzzle"]).replace(".", "0")
        solution = str(ds[i]["solution"])
        rows.append((f"Solve this Sudoku:\n{fmt(puzzle)}", solution))
    return rows


LOADERS = {"gsm8k": load_gsm8k, "strategyqa": load_strategyqa, "sudoku": load_sudoku}


# ── model loading ────────────────────────────────────────────────────────────

def load_model(name, quant, device):
    kwargs = {"trust_remote_code": True}
    if quant in ("fp16", "bf16"):
        kwargs["torch_dtype"] = torch.float16 if quant == "fp16" else torch.bfloat16
        kwargs["device_map"] = device
    elif quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    elif quant == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(name, **kwargs)
    model.eval()
    return model


# ── masked diffusion generation ──────────────────────────────────────────────

def _truncate_at_eos(ids: torch.Tensor, tok) -> torch.Tensor:
    """Truncate generated ids at the first EOS/pad token."""
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    if tok.pad_token_id is not None:
        eos_ids.add(tok.pad_token_id)
    if not eos_ids:
        return ids
    for i, t in enumerate(ids.tolist()):
        if t in eos_ids:
            return ids[:i]
    return ids


def _num_transfer_tokens(mask_num: int, steps: int) -> list[int]:
    """LLaDA get_num_transfer_tokens: distribute mask_num unmasks evenly across steps."""
    base = mask_num // steps
    remainder = mask_num % steps
    return [base + (1 if i < remainder else 0) for i in range(steps)]


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, comp_len: int,
             n_steps: int, mask_id: int, device: str) -> torch.Tensor:
    """
    LLaDA low_confidence remasking (matches generate.py from ML-GSAI/LLaDA).

    At each step:
      - forward pass over full sequence
      - among currently MASKED positions only, pick top-k by confidence
      - unmask those k positions (they stay unmasked for the rest of decoding)

    k is distributed evenly: floor(comp_len/n_steps) per step, with remainder
    spread across the first steps. Once unmasked, a position is never remasked.
    """
    comp = torch.full((comp_len,), mask_id, dtype=torch.long)
    ids = torch.cat([prompt_ids, comp]).unsqueeze(0).to(device)
    prompt_len = len(prompt_ids)

    transfer = _num_transfer_tokens(comp_len, n_steps)

    for step_idx in range(n_steps):
        out = model(ids)
        logits = (out.logits if hasattr(out, "logits") else out[0])[0]

        comp_logits = logits[prompt_len: prompt_len + comp_len].float()
        probs = F.softmax(comp_logits, dim=-1)
        pred = probs.argmax(dim=-1)           # [comp_len]
        conf = probs.max(dim=-1).values       # [comp_len] — same as gather(p, pred)

        # only currently masked positions are candidates
        is_masked = ids[0, prompt_len: prompt_len + comp_len] == mask_id
        confidence = torch.where(is_masked, conf,
                                 torch.full_like(conf, float("-inf")))

        k = min(transfer[step_idx], int(is_masked.sum()))
        if k > 0:
            _, sel = torch.topk(confidence, k=k)
            ids[0, prompt_len + sel] = pred[sel]

    return ids[0, prompt_len: prompt_len + comp_len]


# ── evaluators ───────────────────────────────────────────────────────────────

def _extract_number(text: str):
    """Extract numeric answer: prefer #### format, then 'answer is N', then last number."""
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"(?:answer|result)\s+is\s+([\d,\-\.]+)", text, re.I)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"=\s*\$?([\d,]+)\s*$", text.strip(), re.M)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def eval_gsm8k(generated: str, gold: str) -> dict:
    pred = _extract_number(generated)
    true = _extract_number(gold)
    correct = (pred is not None and true is not None and pred == true)
    return {"correct": int(correct), "pred": pred, "true": true}


def eval_strategyqa(generated: str, gold: str) -> dict:
    gen_lower = generated.lower()
    gold_ans = gold.split(".")[0].strip().lower()  # "yes" or "no"
    if "yes" in gen_lower[:30]:
        pred = "yes"
    elif "no" in gen_lower[:30]:
        pred = "no"
    else:
        pred = None
    correct = (pred == gold_ans)
    return {"correct": int(correct), "pred": pred, "true": gold_ans}


def _is_valid_sudoku(grid_str: str) -> bool:
    """Check if a flat 81-char string is a valid solved sudoku."""
    digits = re.sub(r"[^1-9]", "", grid_str)
    if len(digits) != 81:
        return False
    g = [int(c) for c in digits]
    for i in range(9):
        row = g[i*9:(i+1)*9]
        col = [g[i + j*9] for j in range(9)]
        box_r, box_c = (i // 3) * 3, (i % 3) * 3
        box = [g[(box_r+r)*9 + box_c+c] for r in range(3) for c in range(3)]
        if sorted(row) != list(range(1, 10)): return False
        if sorted(col) != list(range(1, 10)): return False
        if sorted(box) != list(range(1, 10)): return False
    return True


def eval_sudoku(generated: str, gold: str) -> dict:
    valid = _is_valid_sudoku(generated)
    # also check exact match against gold solution
    gen_digits = re.sub(r"[^1-9]", "", generated)
    exact = (gen_digits == gold.replace(".", ""))
    return {"correct": int(exact), "valid": int(valid), "exact": int(exact)}


EVALUATORS = {
    "gsm8k":      eval_gsm8k,
    "strategyqa": eval_strategyqa,
    "sudoku":     eval_sudoku,
}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--fp-quant",  default="fp16")
    parser.add_argument("--q-quant",   default="int4")
    parser.add_argument("--dataset",   default="gsm8k",
                        choices=list(LOADERS))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--comp-len",  type=int, default=512)
    parser.add_argument("--n-steps",   type=int, default=512)
    parser.add_argument("--mask-id",   type=int, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       default="results_quality.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"Model   : {args.model}")
    print(f"Dataset : {args.dataset}  n={args.n_samples}")
    print(f"comp_len={args.comp_len}  n_steps={args.n_steps}")

    print("Loading FP model...")
    fp_model = load_model(args.model, args.fp_quant, device)
    print("Loading Q model...")
    q_model  = load_model(args.model, args.q_quant, device)
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    mask_id = args.mask_id
    if mask_id is None:
        if tok.mask_token is not None:
            mask_id = tok.convert_tokens_to_ids(tok.mask_token)
        else:
            mask_id = 126336
    print(f"Mask ID : {mask_id}")

    pairs = LOADERS[args.dataset](args.n_samples, args.seed)
    evaluator = EVALUATORS[args.dataset]

    fp_results, q_results = [], []
    t0 = time.time()

    for i, (question, gold) in enumerate(pairs):
        prompt_ids = tok(question, return_tensors="pt",
                         add_special_tokens=False)["input_ids"][0]

        fp_gen_ids = generate(fp_model, prompt_ids, args.comp_len,
                              args.n_steps, mask_id, device)
        q_gen_ids  = generate(q_model,  prompt_ids, args.comp_len,
                              args.n_steps, mask_id, device)

        fp_text = tok.decode(_truncate_at_eos(fp_gen_ids, tok), skip_special_tokens=True)
        q_text  = tok.decode(_truncate_at_eos(q_gen_ids,  tok), skip_special_tokens=True)

        fp_eval = evaluator(fp_text, gold)
        q_eval  = evaluator(q_text,  gold)

        fp_results.append({**fp_eval, "generated": fp_text, "gold": gold})
        q_results.append({**q_eval,  "generated": q_text,  "gold": gold})

        if (i + 1) % 10 == 0:
            fp_acc = np.mean([r["correct"] for r in fp_results])
            q_acc  = np.mean([r["correct"] for r in q_results])
            print(f"  {i+1:3d}/{args.n_samples}  ({time.time()-t0:.0f}s)"
                  f"  FP={fp_acc:.3f}  Q={q_acc:.3f}")

    fp_acc = np.mean([r["correct"] for r in fp_results])
    q_acc  = np.mean([r["correct"] for r in q_results])

    print(f"\n{'='*50}")
    print(f"Dataset : {args.dataset}")
    print(f"FP16 accuracy : {fp_acc:.3f}  ({sum(r['correct'] for r in fp_results)}/{len(fp_results)})")
    print(f"INT4 accuracy : {q_acc:.3f}  ({sum(r['correct'] for r in q_results)}/{len(q_results)})")
    print(f"Delta         : {q_acc - fp_acc:+.3f}")

    out = {
        "config": vars(args),
        "fp_accuracy": fp_acc,
        "q_accuracy":  q_acc,
        "delta":       q_acc - fp_acc,
        "fp_results":  fp_results,
        "q_results":   q_results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
