"""The training forward must degrade and select exactly like the decoder does.

Training teaches the student to tolerate a specific corruption. If that
corruption is not the one ``PackedKVCache`` and ``selector_topk`` actually
produce, the student is being fitted to a grid and a selection rule that never
occur at inference, and any recovery it shows would not transfer.

These checks need neither a GPU nor the 7B checkpoint. The end-to-end parity
against the real decoder lives in ``tests/training_parity_e2e.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import SelectorConfig  # noqa: E402
from bitsieve_fastdllm.reference import (  # noqa: E402
    selector_importance_reference,
    simulate_key_quantization,
    simulate_value_quantization,
)
from bitsieve_fastdllm.training.blockdiff import build_masks, select_keep_mask  # noqa: E402
from bitsieve_fastdllm.training.degrade import (  # noqa: E402
    CacheDegradation,
    degrade_keys,
    degrade_values,
)

B, HQ, HKV, D = 1, 28, 4, 128
G = HQ // HKV
BLOCK = 32


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [2, 4])
def test_quant_mode_is_the_cache_grid(bits: int) -> None:
    """mode="quant" must hand back exactly what the packed cache would."""
    torch.manual_seed(0)
    key = torch.randn(B, HKV, 96, D, dtype=torch.bfloat16)
    value = torch.randn(B, HKV, 96, D, dtype=torch.bfloat16)
    cfg = CacheDegradation(mode="quant", k_bits=bits, v_bits=bits)

    assert torch.equal(
        degrade_keys(key, cfg),
        simulate_key_quantization(key, bits=bits, token_group=32, allow_ragged=True),
    )
    assert torch.equal(
        degrade_values(value, cfg),
        simulate_value_quantization(value, bits=bits, channel_group=32),
    )


def test_quant_mode_passes_gradient_through() -> None:
    """The straight-through estimator must leave the gradient untouched."""
    key = torch.randn(B, HKV, 64, D, dtype=torch.float32, requires_grad=True)
    cfg = CacheDegradation(mode="quant", k_bits=4, v_bits=4)
    degrade_keys(key, cfg).sum().backward()
    assert key.grad is not None
    assert torch.allclose(key.grad, torch.ones_like(key.grad))


def test_exact_mode_is_a_no_op() -> None:
    key = torch.randn(B, HKV, 64, D)
    cfg = CacheDegradation(mode="exact")
    assert degrade_keys(key, cfg) is key
    assert degrade_values(key, cfg) is key


def test_noise_variance_matches_the_real_rounding_error() -> None:
    """The Gaussian surrogate's std must track the quantizer's actual error."""
    torch.manual_seed(0)
    key = torch.randn(B, HKV, 1024, D, dtype=torch.float32)
    value = torch.randn(B, HKV, 1024, D, dtype=torch.float32)
    for bits in (2, 4):
        cfg = CacheDegradation(mode="quant", k_bits=bits, v_bits=bits)
        k_err = (degrade_keys(key, cfg) - key).std().item()
        v_err = (degrade_values(value, cfg) - value).std().item()

        noisy = CacheDegradation(mode="noise", k_bits=bits, v_bits=bits)
        g = torch.Generator().manual_seed(0)
        k_sur = (degrade_keys(key, noisy, generator=g) - key).std().item()
        v_sur = (degrade_values(value, noisy, generator=g) - value).std().item()

        assert 0.85 < k_sur / k_err < 1.15, (bits, k_sur, k_err)
        assert 0.85 < v_sur / v_err < 1.15, (bits, v_sur, v_err)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode,queries", [("uniform", 5), ("middle", 1), ("all", 32)])
def test_selection_matches_selector_importance_reference(mode: str, queries: int) -> None:
    """Per-KV-head top-k inside the training forward == the decoder's rule."""
    torch.manual_seed(0)
    n_blocks = 6
    seq = n_blocks * BLOCK
    s = 2 * seq
    dev = "cpu"

    query = torch.randn(B, HQ, s, D)
    key = torch.randn(B, HKV, s, D)
    masks = build_masks(seq, BLOCK, dev, propagate_to_x0=False)

    is_masked = torch.zeros(s, dtype=torch.bool)
    probe_block = 4
    start = probe_block * BLOCK
    masked_rel = [1, 5, 9, 17, 23, 30]
    for r in masked_rel:
        is_masked[start + r] = True

    selector = SelectorConfig(mode=mode, uniform_queries=queries, topk=48, dense_prefix_layers=0)
    keep = select_keep_mask(query, key, masks, is_masked, selector, D**-0.5, G)

    # What the decoder would select for this block, from the same tensors.
    cols = torch.nonzero(masks.old_pair[start], as_tuple=True)[0]
    n_old = int(cols.numel())
    assert n_old == probe_block * BLOCK
    # "all" means all *masked* positions, which is what begin_block() hands the
    # selector - not every position in the block.
    rel = selector.query_indices(masked_rel, BLOCK)
    importance = selector_importance_reference(
        query[:, :, start : start + BLOCK, :],
        key[:, :, cols, :],
        query_indices=rel,
        domain="prefix",
        score_kind="softmax",
        scale=D**-0.5,
    )[0]
    expected = torch.topk(importance, k=selector.effective_topk(n_old), dim=-1).indices

    for h in range(HKV):
        got = torch.nonzero(keep[h, start][cols], as_tuple=True)[0]
        assert set(got.tolist()) == set(expected[h].tolist()), f"head {h}, mode {mode}"


def test_selection_leaves_the_current_block_alone() -> None:
    """Only prefix reads are budgeted; the block attends to itself in full."""
    torch.manual_seed(0)
    seq = 4 * BLOCK
    s = 2 * seq
    query = torch.randn(B, HQ, s, D)
    key = torch.randn(B, HKV, s, D)
    masks = build_masks(seq, BLOCK, "cpu", propagate_to_x0=False)
    is_masked = torch.zeros(s, dtype=torch.bool)
    is_masked[3 * BLOCK : 4 * BLOCK] = True

    selector = SelectorConfig(mode="uniform", uniform_queries=5, topk=16, dense_prefix_layers=0)
    keep = select_keep_mask(query, key, masks, is_masked, selector, D**-0.5, G)

    own = ~masks.old_pair & masks.visible
    assert bool(keep[:, own].all()), "selection must not mask non-prefix reads"


def test_first_block_has_no_prefix_to_budget() -> None:
    torch.manual_seed(0)
    seq = 3 * BLOCK
    query = torch.randn(B, HQ, 2 * seq, D)
    key = torch.randn(B, HKV, 2 * seq, D)
    masks = build_masks(seq, BLOCK, "cpu", propagate_to_x0=False)
    is_masked = torch.zeros(2 * seq, dtype=torch.bool)
    is_masked[:BLOCK] = True

    selector = SelectorConfig(mode="uniform", uniform_queries=5, topk=8, dense_prefix_layers=0)
    keep = select_keep_mask(query, key, masks, is_masked, selector, D**-0.5, G)
    assert bool(keep[:, :BLOCK].all())


def test_budget_above_the_prefix_selects_everything() -> None:
    """A fixed topk larger than the prefix must leave every prefix read open."""
    torch.manual_seed(0)
    seq = 3 * BLOCK
    query = torch.randn(B, HQ, 2 * seq, D)
    key = torch.randn(B, HKV, 2 * seq, D)
    masks = build_masks(seq, BLOCK, "cpu", propagate_to_x0=False)
    is_masked = torch.zeros(2 * seq, dtype=torch.bool)
    is_masked[2 * BLOCK : 3 * BLOCK] = True

    selector = SelectorConfig(mode="uniform", uniform_queries=5, topk=4096, dense_prefix_layers=0)
    keep = select_keep_mask(query, key, masks, is_masked, selector, D**-0.5, G)
    assert bool(keep.all())


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------
def test_masks_reproduce_the_upstream_block_diffusion_mask() -> None:
    """x_t sees its own block plus clean x_0 strictly before it; x_0 is block-causal."""
    seq = 4 * BLOCK
    m = build_masks(seq, BLOCK, "cpu")
    n = seq
    for q in (0, BLOCK + 3, 3 * BLOCK + 31):
        bq = q // BLOCK
        for k in range(2 * n):
            x0_k = k >= n
            bk = (k - n) // BLOCK if x0_k else k // BLOCK
            want = (bq == bk and not x0_k) or (x0_k and bk < bq)
            assert bool(m.visible[q, k]) == want, (q, k)


def test_residual_free_cache_degrades_every_visible_prefix_read() -> None:
    """residual_tokens=0 means no bf16 tail: every prefix read is degraded."""
    seq = 5 * BLOCK
    m = build_masks(seq, BLOCK, "cpu", propagate_to_x0=False)
    xt_prefix_reads = m.visible & m.is_xt[:, None] & ~m.is_xt[None, :]
    assert torch.equal(m.old_pair, xt_prefix_reads)


def test_error_compounds_into_the_cache_by_default() -> None:
    """attend runs before stage_if_committing, so x_0 -> x_0 reads degrade too."""
    seq = 4 * BLOCK
    on = build_masks(seq, BLOCK, "cpu", propagate_to_x0=True)
    off = build_masks(seq, BLOCK, "cpu", propagate_to_x0=False)
    assert int(on.old_pair.sum()) > int(off.old_pair.sum())
    assert torch.equal(on.visible, off.visible)
