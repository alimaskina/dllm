#!/usr/bin/env python3
"""
Experiment 1: Fixed-state fidelity

Core question: does quantization break position confidence ranking (where)
independently from token prediction (what)?

WHAT metrics  — per-position token distribution agreement:
  token_top1/3/5/10_agree   Jaccard overlap of top-k predicted tokens
  kl_fp_q                   KL(p_FP || p_Q) averaged over masked positions

WHERE metrics — cross-position ranking agreement, per confidence signal:
  signal: max_prob (LLaDA default), margin (top1−top2), neg_entropy (Dream)
  for each signal: spearman / kendall / topk_pos_overlap at k=1, k=n//10, k=n//128

Strategies follow literature:
  k=1          Where-to-Unmask paper setup (greedy, cleanest)
  k=n//10      10 denoising steps (common practical default)
  k=n//128     LLaDA default (steps=128), ≈1 for short sequences

Expected signal: token_top1_agree >> topk_pos_overlap_k1[max_prob]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

LLADA_STEPS = 128   # default denoising steps (LLaDA)


def resolve_mask_id(tokenizer, override: int | None) -> int:
    """Return mask token ID: explicit override > tokenizer.mask_token > fallback 126336."""
    if override is not None:
        return override
    if tokenizer.mask_token is not None:
        mid = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        if mid is not None and mid != tokenizer.unk_token_id:
            return mid
    return 126336  # LLaDA default

PROMPTS_GENERIC = [
    "The capital of France is Paris and the Eiffel Tower was built in 1889 as",
    "To solve a quadratic equation ax squared plus bx plus c equals zero we use",
    "The quick brown fox jumps over the lazy dog which is a pangram containing",
    "In machine learning gradient descent is an optimization algorithm that iteratively",
    "Water molecules consist of two hydrogen atoms and one oxygen atom bonded together",
    "Shakespeare wrote many famous plays including Hamlet Macbeth and Romeo and Juliet in",
    "The human body contains approximately 37 trillion cells and the brain processes information",
    "Python is a high level programming language known for its readable syntax and extensive",
    "The theory of relativity was developed by Albert Einstein and changed our understanding",
    "Climate change refers to long term shifts in global temperatures and weather patterns that",
    "The speed of light in a vacuum is approximately 299792458 meters per second which",
    "Neural networks are computational models loosely inspired by the structure of biological",
    "The French Revolution began in 1789 and resulted in the abolition of the monarchy",
    "Photosynthesis is the process by which plants use sunlight water and carbon dioxide to",
    "The periodic table organizes chemical elements by atomic number and groups elements with",
]

PROMPTS_DIVERSE = [
    # math reasoning
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest for $2 each. How much does she make per day?",
    "A store sells notebooks for $3 each and pens for $1.50 each. Maria bought 4 notebooks and 6 pens. What is the total cost?",
    "There are 5 boxes each containing 12 apples. If 7 apples are removed from each box, how many apples remain in total?",
    "A train travels 240 miles in 4 hours. At this rate, how far will it travel in 7 hours?",
    "If x plus 5 equals 17, then x minus 3 equals what?",
    "A rectangle has length 8 cm and width 5 cm. What is the area and the perimeter?",
    # code / algorithms
    "To reverse a linked list in Python we iterate through the nodes and reassign pointers so that",
    "Binary search works by repeatedly dividing the search interval in half and comparing the target to",
    "The time complexity of merge sort is O(n log n) because at each level of recursion we",
    "def fibonacci(n): if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2). This function",
    "A hash table achieves O(1) average lookup time by computing a hash function that maps keys to",
    "The quicksort algorithm selects a pivot element and partitions the array into elements less than and",
    # logical / commonsense reasoning
    "All mammals are warm-blooded. Whales are mammals. Therefore we can conclude that whales are",
    "If it is raining then the ground is wet. The ground is not wet. Therefore it follows that",
    "John is taller than Mary. Mary is taller than Bob. Who is the shortest and how do we know",
    "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. The ball costs",
    "Three friends split a restaurant bill equally. The total was $48 plus a 15% tip. Each person pays",
    "If today is Wednesday and the meeting is in 10 days, then the meeting falls on a",
    # science
    "Newton's second law states that force equals mass times acceleration which means that if we double",
    "The mitochondria produces ATP through cellular respiration by breaking down glucose in a process called",
    "DNA replication is semiconservative meaning that each new double helix contains one original strand and",
    "The ideal gas law PV equals nRT relates pressure volume and temperature of a gas. If we double",
    "An atom of carbon has 6 protons 6 neutrons and 6 electrons. When it forms CO2 it shares electrons with",
    "Ohm's law states that voltage equals current times resistance. In a circuit with 12V and 4 ohms the",
    # formal / structured
    "The defendant is hereby ordered to pay damages in the amount of five thousand dollars within thirty",
    "In order to install the software first download the installer then run it as administrator and follow",
    "The patient presents with a fever of 38.5 degrees Celsius elevated white blood cell count and",
    "Article 1 Section 2 of the Constitution provides that representatives shall be apportioned among the states according",
    "To prepare the solution dissolve 5 grams of sodium chloride in 100 milliliters of distilled water and",
    "The shareholders hereby authorize the board of directors to issue up to one million additional shares of",
]


def load_gsm8k_prompts(n: int = 200, seed: int = 42) -> list[str]:
    """Load question strings from GSM8K train split."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
        return [ds[i]["question"] for i in indices]
    except Exception as e:
        print(f"Warning: could not load GSM8K ({e}), falling back to generic prompts.")
        return PROMPTS_GENERIC


def load_gsm8k_pairs(n: int = 200, seed: int = 42) -> list[tuple[str, str]]:
    """Load (question, answer) pairs from GSM8K train split for gold-teacher states."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
        return [(ds[i]["question"], ds[i]["answer"]) for i in indices]
    except Exception as e:
        print(f"Warning: could not load GSM8K ({e}).")
        return [(p, "") for p in PROMPTS_GENERIC]


def load_wikitext_pairs(n: int = 200, seed: int = 42) -> list[tuple[str, str]]:
    """WikiText-103: split each long paragraph ~30/70 at first sentence boundary."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 300]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(texts), min(n, len(texts)), replace=False).tolist()
        pairs = []
        for i in idx:
            t = texts[i]
            cut = t.find(". ", int(len(t) * 0.25))
            cut = cut + 1 if cut != -1 else int(len(t) * 0.3)
            pairs.append((t[:cut].strip(), t[cut:].strip()))
        return pairs
    except Exception as e:
        print(f"Warning: could not load WikiText-103 ({e}).")
        return [(p, "") for p in PROMPTS_GENERIC]


def load_humaneval_pairs(n: int = 164, seed: int = 42) -> list[tuple[str, str]]:
    """HumanEval: function docstring prompt → canonical solution."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
        return [(ds[i]["prompt"], ds[i]["canonical_solution"]) for i in idx]
    except Exception as e:
        print(f"Warning: could not load HumanEval ({e}).")
        return [(p, "") for p in PROMPTS_GENERIC]


def load_mbpp_pairs(n: int = 200, seed: int = 42) -> list[tuple[str, str]]:
    """MBPP: task description → code solution."""
    try:
        from datasets import load_dataset
        ds = load_dataset("mbpp", split="test")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
        return [(ds[i]["text"], ds[i]["code"]) for i in idx]
    except Exception as e:
        print(f"Warning: could not load MBPP ({e}).")
        return [(p, "") for p in PROMPTS_GENERIC]


def load_strategyqa_pairs(n: int = 200, seed: int = 42) -> list[tuple[str, str]]:
    """StrategyQA: question → yes/no + supporting facts."""
    try:
        from datasets import load_dataset
        ds = load_dataset("metaeval/strategyqa", split="train")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()
        pairs = []
        for i in idx:
            row = ds[i]
            ans = "Yes" if row["answer"] else "No"
            facts = row.get("facts", [])
            completion = ans + ". " + " ".join(facts) if facts else ans
            pairs.append((row["question"], completion))
        return pairs
    except Exception as e:
        print(f"Warning: could not load StrategyQA ({e}).")
        return [(p, "") for p in PROMPTS_GENERIC]


def load_sudoku_pairs(n: int = 200, seed: int = 42) -> list[tuple[str, str]]:
    """Sudoku 9×9: puzzle grid (with 0s for blanks) → solution grid."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Ritvik19/sudoku-dataset", split="train")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False).tolist()

        def fmt(s81):
            return "\n".join(" ".join(s81[r*9:(r+1)*9]) for r in range(9))

        # column names vary by dataset version
        puzzle_col   = "puzzle"
        solution_col = "solution"

        pairs = []
        for i in idx:
            puzzle   = str(ds[i][puzzle_col]).replace(".", "0")
            solution = str(ds[i][solution_col])
            pairs.append((f"Solve this Sudoku:\n{fmt(puzzle)}", fmt(solution)))
        return pairs
    except Exception as e:
        print(f"Warning: could not load Sudoku ({e}).")
        return [(p, "") for p in PROMPTS_GENERIC]


def load_pairs(dataset: str, n: int = 200, seed: int = 42) -> list[tuple[str, str]]:
    loaders = {
        "gsm8k":      load_gsm8k_pairs,
        "wikitext":   load_wikitext_pairs,
        "humaneval":  load_humaneval_pairs,
        "mbpp":       load_mbpp_pairs,
        "strategyqa": load_strategyqa_pairs,
        "sudoku":     load_sudoku_pairs,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {list(loaders)}")
    return loaders[dataset](n=n, seed=seed)


def get_prompts(prompt_set: str, n: int = 200, seed: int = 42) -> list[str]:
    if prompt_set == "gsm8k":
        return load_gsm8k_prompts(n, seed)
    elif prompt_set == "diverse":
        return PROMPTS_DIVERSE
    else:
        return PROMPTS_GENERIC

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, quant: str, device: str):
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
    else:
        raise ValueError(f"Unknown quant: {quant}")

    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Forward pass — returns all signals needed for what + where metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_masked(model, input_ids: torch.Tensor, masked_pos: list[int]):
    """
    Single forward pass over masked positions.
    Returns a dict of per-position signals (all on CPU):
      pred_tokens  [n]   top-1 predicted token id
      confidence   [n]   max(softmax) — LLaDA/MaskGIT confidence
      margin       [n]   top1_prob − top2_prob — Dream topk_margin signal
      neg_entropy  [n]   −H(p) — Dream entropy signal (higher = more confident)
      top10_idx    [n,10] top-10 token indices by probability
      log_probs    [n, V] full log-softmax (kept briefly for KL, then discarded)
    """
    output = model(input_ids)
    logits = output.logits if hasattr(output, "logits") else output[0]
    logits = logits[0]  # [seq_len, vocab]

    idx = torch.tensor(masked_pos, device=logits.device, dtype=torch.long)
    pos_logits = logits[idx].float()          # [n, vocab]

    log_probs = F.log_softmax(pos_logits, dim=-1)
    probs = log_probs.exp()

    top2 = probs.topk(2, dim=-1)
    confidence = top2.values[:, 0]
    pred_tokens = top2.indices[:, 0]
    margin = top2.values[:, 0] - top2.values[:, 1]

    entropy = -(probs * log_probs).sum(dim=-1)
    neg_entropy = -entropy

    top10_idx = probs.topk(10, dim=-1).indices

    return {
        "pred_tokens": pred_tokens.cpu(),
        "confidence":  confidence.cpu(),
        "margin":      margin.cpu(),
        "neg_entropy": neg_entropy.cpu(),
        "top10_idx":   top10_idx.cpu(),
        "log_probs":   log_probs.cpu(),
    }


# ---------------------------------------------------------------------------
# State generation
# ---------------------------------------------------------------------------

def make_masked_states(
    prompt_ids: torch.Tensor,
    comp_len: int,
    device: str,
    mask_ratios: list[float],
    rng: np.random.Generator,
    mask_id: int = 126336,
) -> list[dict]:
    states = []
    prompt_np = prompt_ids.numpy()

    for mask_ratio in mask_ratios:
        n_masked = max(1, int(comp_len * mask_ratio))
        masked_rel = rng.choice(comp_len, n_masked, replace=False).tolist()
        masked_set = set(masked_rel)

        comp = np.zeros(comp_len, dtype=np.int64)
        for pos in range(comp_len):
            if pos in masked_set:
                comp[pos] = mask_id
            else:
                comp[pos] = int(prompt_np[pos % len(prompt_np)])

        full_ids = torch.cat([
            prompt_ids,
            torch.from_numpy(comp),
        ]).unsqueeze(0).to(device)

        global_masked = [len(prompt_ids) + r for r in sorted(masked_rel)]
        label = "early" if mask_ratio >= 0.7 else ("mid" if mask_ratio >= 0.3 else "late")

        states.append({
            "input_ids":      full_ids,
            "masked_positions": global_masked,
            "mask_ratio":     mask_ratio,
            "label":          label,
        })

    return states


def make_gold_states(
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    comp_len: int,
    device: str,
    mask_ratios: list[float],
    rng: np.random.Generator,
    mask_id: int = 126336,
) -> list[dict]:
    """
    Gold-teacher states: revealed positions contain actual gold completion tokens.
    This matches teacher-forced denoising — the revealed context is always correct,
    so confidence geometry is not distorted by out-of-distribution tokens.
    """
    # Truncate to comp_len; use however many gold tokens we have (no padding)
    if len(completion_ids) > comp_len:
        completion_ids = completion_ids[:comp_len]
    effective_len = len(completion_ids)

    states = []
    comp_np = completion_ids.numpy().astype(np.int64)

    for mask_ratio in mask_ratios:
        n_masked = max(1, int(effective_len * mask_ratio))
        masked_rel = rng.choice(effective_len, n_masked, replace=False).tolist()
        masked_set = set(masked_rel)

        comp = comp_np.copy()
        for pos in masked_set:
            comp[pos] = mask_id

        full_ids = torch.cat([
            prompt_ids,
            torch.from_numpy(comp),
        ]).unsqueeze(0).to(device)

        global_masked = [len(prompt_ids) + r for r in sorted(masked_rel)]
        label = "early" if mask_ratio >= 0.7 else ("mid" if mask_ratio >= 0.3 else "late")

        states.append({
            "input_ids":        full_ids,
            "masked_positions": global_masked,
            "mask_ratio":       mask_ratio,
            "label":            label,
        })

    return states


# ---------------------------------------------------------------------------
# WHAT metrics — token distribution agreement
# ---------------------------------------------------------------------------

def compute_what_metrics(
    fp_out: dict,
    q_out:  dict,
    token_k_list: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Per-position token distribution comparison, averaged over positions."""
    fp_pred   = fp_out["pred_tokens"]
    q_pred    = q_out["pred_tokens"]
    fp_top10  = fp_out["top10_idx"]   # [n, 10]
    q_top10   = q_out["top10_idx"]
    fp_logp   = fp_out["log_probs"]   # [n, vocab]
    q_logp    = q_out["log_probs"]

    n = len(fp_pred)
    out = {}

    # top-k token Jaccard overlap per position, then averaged
    for k in token_k_list:
        if k > 10:
            continue  # we only stored top-10
        overlaps = []
        for i in range(n):
            a = set(fp_top10[i, :k].tolist())
            b = set(q_top10[i, :k].tolist())
            overlaps.append(len(a & b) / k)
        out[f"token_top{k}_agree"] = float(np.mean(overlaps))

    # KL(p_FP || p_Q) per position, averaged
    # KL = sum(p_FP * (log_p_FP - log_p_Q))
    p_fp = fp_logp.exp()
    kl_per_pos = (p_fp * (fp_logp - q_logp)).sum(dim=-1)  # [n]
    out["kl_fp_q"] = kl_per_pos.mean().item()

    return out


# ---------------------------------------------------------------------------
# WHERE metrics — cross-position ranking per confidence signal
# ---------------------------------------------------------------------------

def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, len(a))
    sa = set(a.topk(k).indices.tolist())
    sb = set(b.topk(k).indices.tolist())
    return len(sa & sb) / k


def compute_where_metrics(
    fp_signal: torch.Tensor,
    q_signal:  torch.Tensor,
    signal_name: str,
) -> dict:
    """
    Cross-position ranking metrics for one confidence signal.

    topk overlaps (k values from literature):
      k=1      Where-to-Unmask paper (greedy, most sensitive)
      k=n//10  10 denoising steps
      k=n//128 LLaDA default (steps=128)

    boundary_margin: c_(1) - c_(2) in FP signal — how close the top-2 positions
      are. Small margin → quantization noise easily flips the boundary decision.

    rank_displacement: rank Q assigns to FP's argmax position (1 = perfect match).
      Tells us not just whether the top position changed, but how far it moved.
    """
    n = len(fp_signal)
    pre = signal_name
    out = {}

    k_greedy  = 1
    k_10step  = max(1, n // 10)
    k_llada   = max(1, n // LLADA_STEPS)

    out[f"{pre}_topk1_overlap"]       = _topk_overlap(fp_signal, q_signal, k_greedy)
    out[f"{pre}_topk10step_overlap"]  = _topk_overlap(fp_signal, q_signal, k_10step)
    out[f"{pre}_topkllada_overlap"]   = _topk_overlap(fp_signal, q_signal, k_llada)

    # Boundary margin (FP): gap between top-1 and top-2 confidence across positions
    if n >= 2:
        fp_top2 = fp_signal.topk(2).values
        out[f"{pre}_boundary_margin"] = (fp_top2[0] - fp_top2[1]).item()

    # Rank displacement: what rank does Q assign to FP's top-1 position?
    fp_top1_idx = fp_signal.argmax().item()
    q_order = q_signal.argsort(descending=True).tolist()
    out[f"{pre}_rank_displacement"] = q_order.index(fp_top1_idx) + 1  # 1-indexed

    return out


# ---------------------------------------------------------------------------
# Full metrics for one (state, fp_out, q_out) triple
# ---------------------------------------------------------------------------

def compute_metrics(fp_out: dict, q_out: dict) -> dict:
    what = compute_what_metrics(fp_out, q_out)

    where = {}
    for signal_name in ("confidence", "margin", "neg_entropy"):
        where.update(compute_where_metrics(
            fp_out[signal_name],
            q_out[signal_name],
            signal_name,
        ))

    # Key diagnostic: token top-1 vs greedy-k=1 position agreement on LLaDA signal
    gap = what.get("token_top1_agree", float("nan")) - where.get("confidence_topk1_overlap", float("nan"))

    return {**what, **where, "what_where_gap": gap}


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict]) -> dict:
    skip = {"n_masked", "label", "mask_ratio", "sample"}
    keys = [k for k in rows[0] if k not in skip]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))
    return agg


# Report order: most interpretable metrics first
REPORT_KEYS = [
    # WHAT
    ("token_top1_agree",              "token top-1 agree           [WHAT]"),
    ("token_top3_agree",              "token top-3 agree           [WHAT]"),
    ("token_top5_agree",              "token top-5 agree           [WHAT]"),
    ("token_top10_agree",             "token top-10 agree          [WHAT]"),
    ("kl_fp_q",                       "KL(FP||Q) per position      [WHAT]"),
    # WHERE — confidence (LLaDA / MaskGIT)
    ("confidence_topk1_overlap",      "conf topk-1 overlap         [WHERE/conf]"),
    ("confidence_topk10step_overlap", "conf topk-10step overlap    [WHERE/conf]"),
    ("confidence_topkllada_overlap",  "conf topk-llada overlap     [WHERE/conf]"),
    ("confidence_boundary_margin",    "conf boundary margin        [WHERE/conf]"),
    ("confidence_rank_displacement",  "conf rank displacement      [WHERE/conf]"),
    # WHERE — margin (Dream topk_margin)
    ("margin_topk1_overlap",          "margin topk-1 overlap       [WHERE/margin]"),
    ("margin_topk10step_overlap",     "margin topk-10step overlap  [WHERE/margin]"),
    ("margin_boundary_margin",        "margin boundary margin      [WHERE/margin]"),
    ("margin_rank_displacement",      "margin rank displacement    [WHERE/margin]"),
    # WHERE — neg_entropy (Dream entropy)
    ("neg_entropy_topk1_overlap",     "neg-entropy topk-1 overlap  [WHERE/entropy]"),
    ("neg_entropy_topk10step_overlap","neg-entropy topk-10step     [WHERE/entropy]"),
    ("neg_entropy_boundary_margin",   "neg-entropy boundary margin [WHERE/entropy]"),
    ("neg_entropy_rank_displacement", "neg-entropy rank displace   [WHERE/entropy]"),
    # SIGNAL
    ("what_where_gap",                "what − where gap            [SIGNAL]"),
]


def print_section(label: str, agg: dict):
    print(f"\n  ── {label} ──")
    for key, desc in REPORT_KEYS:
        mk, sk = f"{key}_mean", f"{key}_std"
        if mk in agg:
            print(f"    {desc:45s}  {agg[mk]:+.4f} ± {agg[sk]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp1: Fixed-state fidelity — what vs where under quantization"
    )
    parser.add_argument("--model",       default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--fp-quant",    default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--q-quant",     default="int4", choices=["int8", "int4"])
    parser.add_argument("--q-model",     default=None,
                        help="Separate checkpoint for quantized model (optional)")
    parser.add_argument("--comp-len",    type=int, default=128)
    parser.add_argument("--mask-ratios", nargs="+", type=float, default=[0.9, 0.6, 0.2],
                        help="early / mid / late decoding stages")
    parser.add_argument("--n-samples",   type=int, default=200)
    parser.add_argument("--prompt-set",  default="generic",
                        choices=["generic", "diverse", "gsm8k"],
                        help="Which prompt set to use")
    parser.add_argument("--mask-id",     type=int, default=None,
                        help="Mask token ID (auto-detected from tokenizer if not set)")
    parser.add_argument("--state-type",  default="synthetic",
                        choices=["synthetic", "gold"],
                        help="synthetic: fill revealed positions with cycled prompt tokens; "
                             "gold: use ground-truth completion tokens")
    parser.add_argument("--dataset",     default="gsm8k",
                        choices=["gsm8k", "wikitext", "humaneval", "mbpp", "strategyqa", "sudoku"],
                        help="Dataset for gold-teacher states (used when --state-type=gold)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out",         default="results_exp1.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    q_name = args.q_model or args.model

    print(f"Device : {device}")
    print(f"FP     : {args.model} [{args.fp_quant}]")
    print(f"Q      : {q_name} [{args.q_quant}]")
    print(f"Samples: {args.n_samples} × {len(args.mask_ratios)} = "
          f"{args.n_samples * len(args.mask_ratios)} states")

    print("Loading FP model...")
    fp_model = load_model(args.model, args.fp_quant, device)
    print("Loading Q model...")
    q_model  = load_model(q_name, args.q_quant, device)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    mask_id = resolve_mask_id(tokenizer, args.mask_id)
    print(f"Mask token ID : {mask_id}  ({tokenizer.convert_ids_to_tokens(mask_id)})")
    print(f"State type    : {args.state_type!r}")

    completion_ids_list = None
    if args.state_type == "gold":
        pairs = load_pairs(args.dataset, n=args.n_samples, seed=args.seed)
        print(f"Dataset       : {args.dataset!r} (gold pairs, {len(pairs)} samples)")
        prompt_ids_list = []
        completion_ids_list = []
        for q, a in pairs:
            q_ids = tokenizer(q, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            a_ids = tokenizer(a, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            prompt_ids_list.append(q_ids)
            completion_ids_list.append(a_ids)
    else:
        prompts = get_prompts(args.prompt_set, n=args.n_samples, seed=args.seed)
        print(f"Prompt set    : {args.prompt_set!r}  ({len(prompts)} unique prompts)")
        prompt_ids_list = [
            tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            for p in prompts
        ]

    all_results: list[dict] = []
    per_label: dict[str, list[dict]] = {"early": [], "mid": [], "late": []}

    print("\nRunning forward passes...")
    t0 = time.time()

    for i in range(args.n_samples):
        prompt_ids = prompt_ids_list[i % len(prompt_ids_list)]
        if args.state_type == "gold" and completion_ids_list is not None:
            comp_ids = completion_ids_list[i % len(completion_ids_list)]
            states = make_gold_states(
                prompt_ids, comp_ids, args.comp_len, device, args.mask_ratios, rng, mask_id
            )
        else:
            states = make_masked_states(
                prompt_ids, args.comp_len, device, args.mask_ratios, rng, mask_id
            )

        for state in states:
            fp_out = forward_masked(fp_model, state["input_ids"], state["masked_positions"])
            q_out  = forward_masked(q_model,  state["input_ids"], state["masked_positions"])

            metrics = compute_metrics(fp_out, q_out)
            metrics["sample"]     = i
            metrics["mask_ratio"] = state["mask_ratio"]
            metrics["label"]      = state["label"]
            metrics["n_masked"]   = len(state["masked_positions"])

            # free log_probs immediately after KL is computed
            del fp_out["log_probs"], q_out["log_probs"]

            all_results.append(metrics)
            per_label[state["label"]].append(metrics)

        if (i + 1) % 5 == 0:
            print(f"  {i+1:3d}/{args.n_samples}  ({time.time()-t0:.1f}s)")

    print(f"\nDone in {time.time()-t0:.1f}s")

    print("\n" + "=" * 65)
    print("EXPERIMENT 1 — FIXED-STATE FIDELITY")
    print(f"FP: {args.model} [{args.fp_quant}]  vs  Q: {q_name} [{args.q_quant}]")
    print("Hypothesis: token_top1_agree >> confidence_topk1_overlap")
    print("=" * 65)

    summary: dict = {}
    for label in ["early", "mid", "late"]:
        if not per_label[label]:
            continue
        agg = aggregate(per_label[label])
        summary[label] = agg
        print_section(label, agg)

    agg_all = aggregate(all_results)
    summary["all"] = agg_all
    print_section("ALL", agg_all)

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "fp_model":    args.model,
                "fp_quant":    args.fp_quant,
                "q_model":     q_name,
                "q_quant":     args.q_quant,
                "n_samples":   args.n_samples,
                "prompt_set":  args.prompt_set,
                "state_type":  args.state_type,
                "dataset":     args.dataset,
                "mask_id":     mask_id,
                "comp_len":    args.comp_len,
                "mask_ratios": args.mask_ratios,
                "llada_steps": LLADA_STEPS,
                "seed":        args.seed,
            },
            "summary": summary,
            "samples": all_results,
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
