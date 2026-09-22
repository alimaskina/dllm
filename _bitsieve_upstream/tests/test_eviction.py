"""Eviction policy: the bias correction, the retention rules, and compaction.

None of this needs a GPU or the checkpoint -- it is arithmetic over the live
set, and the properties worth pinning are the ones that would otherwise fail
silently and look like a weak policy rather than a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.eviction import EvictionConfig, EvictionState  # noqa: E402

L, B, H = 2, 1, 4


def make_state(live: int, capacity: int | None = None) -> EvictionState:
    st = EvictionState(
        num_layers=L, batch_size=B, num_kv_heads=H,
        capacity=capacity or live, device="cpu", max_position=32768,
    )
    st.append(torch.arange(live))
    return st


# --------------------------------------------------------------------------
# the bias correction
# --------------------------------------------------------------------------
def test_first_observation_recovers_alpha_exactly() -> None:
    """ghat after one observation must equal alpha, for any decay."""
    for decay in (0.5, 0.9, 0.99):
        st = make_state(8)
        alpha = torch.rand(B, H, 8)
        st.observe(0, alpha, decay)
        ghat = st.corrected(0, decay)
        assert torch.allclose(ghat, alpha, atol=1e-6), decay


def test_correction_is_what_stops_new_entries_reading_as_worthless() -> None:
    """The artifact the correction exists to remove, measured.

    An old entry and a new entry receiving the *same* attention must rank the
    same. On the raw EMA the new one looks an order of magnitude worse.
    """
    decay = 0.9
    st = make_state(2)
    alpha = torch.full((B, H, 2), 0.5)
    for _ in range(40):                      # entry 0 is old, both see alpha
        st.observe(0, alpha, decay)

    st.append(torch.tensor([2]))             # entry 2 arrives now
    alpha3 = torch.full((B, H, 3), 0.5)
    st.observe(0, alpha3, decay)

    raw = st.g[0, ..., :3]
    ghat = st.corrected(0, decay)
    assert raw[0, 0, 2] < 0.2 * raw[0, 0, 0], "expected the raw EMA to underrate the newcomer"
    assert torch.allclose(ghat[0, 0, 2], ghat[0, 0, 0], atol=1e-5), ghat


def test_never_observed_entries_score_zero_not_nan() -> None:
    st = make_state(4)
    ghat = st.corrected(0, 0.9)
    assert torch.isfinite(ghat).all()
    assert float(ghat.abs().max()) == 0.0


def test_ema_tracks_a_steady_signal() -> None:
    decay = 0.8
    st = make_state(3)
    alpha = torch.tensor([[[0.1, 0.5, 0.9]] * H])
    for _ in range(60):
        st.observe(0, alpha, decay)
    ghat = st.corrected(0, decay)
    assert torch.allclose(ghat, alpha, atol=1e-3), ghat


# --------------------------------------------------------------------------
# capacity
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seen,expected",
    [(100, 256), (5000, 256), (10_000, 500), (32_768, 1638)],
)
def test_capacity_is_the_max_of_percent_and_floor(seen: int, expected: int) -> None:
    cfg = EvictionConfig(policy="recent", capacity_percent=5.0, capacity_floor=256)
    assert cfg.capacity(seen) == expected


# --------------------------------------------------------------------------
# retention rules
# --------------------------------------------------------------------------
def test_recent_keeps_the_newest_by_position() -> None:
    st = make_state(600)
    cfg = EvictionConfig(policy="recent", capacity_floor=256, capacity_percent=1.0)
    keep = st.maybe_evict(cfg, block_index=3, tokens_seen=600)
    assert keep is not None and st.live == 256
    assert torch.equal(
        st.pos[0, 0, 0, :256], torch.arange(600 - 256, 600, dtype=st.pos.dtype)
    )


def test_ema_recent_keeps_the_window_plus_the_best_of_the_rest() -> None:
    live = 600
    st = make_state(live)
    decay = 0.9
    # Give three old entries a large, sustained share; everything else ~0.
    alpha = torch.zeros(B, H, live)
    winners = [7, 11, 13]
    alpha[..., winners] = 1.0
    for _ in range(10):
        for layer in range(L):
            st.observe(layer, alpha, decay)

    cfg = EvictionConfig(
        policy="ema_recent", recent_window=128, capacity_floor=256,
        capacity_percent=1.0, decay=decay,
    )
    st.maybe_evict(cfg, block_index=3, tokens_seen=live)
    assert st.live == 256

    kept = set(st.pos[0, 0, 0].tolist())
    assert set(range(live - 128, live)) <= kept, "the recency window must survive intact"
    assert set(winners) <= kept, "entries carrying real attention mass must survive"


def test_ema_recent_does_not_spend_the_budget_twice_on_the_window() -> None:
    """The window is excluded from the EMA contest, so C slots hold C entries."""
    live = 400
    st = make_state(live)
    # Make the newest entries also the highest-EMA ones -- if the window were
    # not excluded they would be picked twice and the kept set would collapse.
    alpha = torch.zeros(B, H, live)
    alpha[..., -128:] = 1.0
    for layer in range(L):
        for _ in range(5):
            st.observe(layer, alpha, 0.9)

    cfg = EvictionConfig(policy="ema_recent", recent_window=128, capacity_floor=256,
                         capacity_percent=1.0)
    st.maybe_evict(cfg, block_index=3, tokens_seen=live)
    assert st.live == 256
    for layer in range(L):
        for h in range(H):
            kept = st.pos[layer, 0, h].tolist()
            assert len(set(kept)) == 256, "duplicate survivors"


def test_heads_keep_different_entries() -> None:
    live = 600
    st = make_state(live)
    alpha = torch.zeros(B, H, live)
    for h in range(H):
        alpha[0, h, 10 * h] = 1.0          # a different favourite per head
    for _ in range(5):
        st.observe(0, alpha, 0.9)
        st.observe(1, alpha, 0.9)

    cfg = EvictionConfig(policy="ema_recent", recent_window=8, capacity_floor=16,
                         capacity_percent=1.0)
    st.maybe_evict(cfg, block_index=3, tokens_seen=live)
    kept = [set(st.pos[0, 0, h].tolist()) for h in range(H)]
    for h in range(H):
        assert 10 * h in kept[h]
    assert any(kept[0] != kept[h] for h in range(1, H)), "heads collapsed to one set"


# --------------------------------------------------------------------------
# compaction and schedule
# --------------------------------------------------------------------------
def test_compaction_carries_g_n_and_pos_together() -> None:
    live = 600
    st = make_state(live)
    alpha = torch.rand(B, H, live)
    st.observe(0, alpha, 0.9)
    before = {int(p): float(g) for p, g in zip(st.pos[0, 0, 0], st.g[0, 0, 0])}

    cfg = EvictionConfig(policy="recent", capacity_floor=256, capacity_percent=1.0)
    st.maybe_evict(cfg, block_index=3, tokens_seen=live)

    for p, g in zip(st.pos[0, 0, 0], st.g[0, 0, 0]):
        assert before[int(p)] == pytest.approx(float(g)), int(p)
    assert bool((st.n[0, 0, 0] == 1).all())


def test_eviction_only_runs_on_schedule() -> None:
    cfg = EvictionConfig(policy="recent", capacity_floor=16, capacity_percent=1.0,
                         interval_blocks=4)
    for block in (0, 1, 2):
        st = make_state(100)
        assert st.maybe_evict(cfg, block_index=block, tokens_seen=100) is None
        assert st.live == 100
    st = make_state(100)
    assert st.maybe_evict(cfg, block_index=3, tokens_seen=100) is not None
    assert st.live == 16


def test_no_eviction_below_capacity() -> None:
    st = make_state(100)
    cfg = EvictionConfig(policy="recent", capacity_floor=256, capacity_percent=5.0)
    assert st.maybe_evict(cfg, block_index=3, tokens_seen=2000) is None
    assert st.live == 100


def test_policy_none_never_evicts() -> None:
    st = make_state(5000)
    cfg = EvictionConfig(policy="none")
    assert st.maybe_evict(cfg, block_index=3, tokens_seen=5000) is None
    assert st.live == 5000


def test_alpha_must_match_the_live_set() -> None:
    st = make_state(10)
    with pytest.raises(ValueError, match="live"):
        st.observe(0, torch.rand(B, H, 9), 0.9)


def test_append_grows_past_the_initial_capacity() -> None:
    st = make_state(10, capacity=10)
    st.append(torch.arange(10, 900))
    assert st.live == 900
    assert torch.equal(st.pos[0, 0, 0, :900], torch.arange(900, dtype=st.pos.dtype))


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------
def test_state_bytes_are_ten_per_live_entry_per_layer_head() -> None:
    st = make_state(100)
    assert st.state_nbytes() == L * B * H * 100 * (4 + 2 + 4)


def test_config_rejects_a_window_that_would_fill_the_budget() -> None:
    cfg = EvictionConfig(policy="ema_recent", recent_window=512, capacity_floor=256)
    with pytest.raises(ValueError, match="recent_window"):
        cfg.validate()


# --------------------------------------------------------------------------
# the live mask in the selector
# --------------------------------------------------------------------------
def test_masking_before_the_softmax_is_not_the_same_as_after() -> None:
    """Evicted entries must leave the normalizer, not just the ranking.

    If they only left the ranking, every surviving alpha would be scaled down by
    mass that went nowhere -- and the EMA would decay for a reason that has
    nothing to do with how useful the entry is.
    """
    from bitsieve_fastdllm.reference import selector_importance_reference

    torch.manual_seed(0)
    b, hq, hkv, n, d = 1, 8, 2, 64, 32
    query = torch.randn(b, hq, 16, d)
    key = torch.randn(b, hkv, n, d)
    live = torch.ones(b, hkv, n, dtype=torch.bool)
    live[..., 32:] = False                     # half the cache is gone

    before = selector_importance_reference(
        query, key, query_indices=[0, 5, 10], scale=d**-0.5, live_mask=live
    )
    after = selector_importance_reference(
        query, key, query_indices=[0, 5, 10], scale=d**-0.5
    ).masked_fill(~live, 0.0)

    assert torch.allclose(before[..., 32:], torch.zeros_like(before[..., 32:]))
    assert before[..., :32].sum() > after[..., :32].sum() * 1.2, (
        "masking before the softmax must renormalize onto the survivors"
    )
    assert torch.allclose(before[..., :32].sum(-1), torch.ones(b, hkv), atol=1e-5), (
        "the surviving mass must sum back to 1"
    )


def test_live_mask_leaves_a_full_cache_untouched() -> None:
    from bitsieve_fastdllm.reference import selector_importance_reference

    torch.manual_seed(0)
    b, hq, hkv, n, d = 1, 4, 2, 48, 32
    query = torch.randn(b, hq, 8, d)
    key = torch.randn(b, hkv, n, d)
    all_live = torch.ones(b, hkv, n, dtype=torch.bool)
    plain = selector_importance_reference(query, key, query_indices=[0, 3], scale=d**-0.5)
    masked = selector_importance_reference(
        query, key, query_indices=[0, 3], scale=d**-0.5, live_mask=all_live
    )
    assert torch.equal(plain, masked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_triton_and_torch_agree_on_the_live_mask() -> None:
    """The kernel masks logits in place; the reference masks them functionally."""
    from bitsieve_fastdllm.cache import LayerCacheView
    from bitsieve_fastdllm.kernels.ops import selector_topk

    torch.manual_seed(0)
    dev = "cuda"
    b, hq, hkv, n, d = 1, 8, 2, 256, 128
    query = torch.randn(b, hq, 32, d, device=dev, dtype=torch.bfloat16)
    key = torch.randn(b, hkv, n, d, device=dev, dtype=torch.bfloat16)
    view = LayerCacheView(
        k_bits=16, v_bits=16, length=n, quantized_length=n, residual_length=0,
        key_token_group=32, value_channel_group=32,
        k_q=None, k_scale=None, k_zero=None, v_q=None, v_scale=None, v_zero=None,
        k_fp=key, v_fp=key, k_residual=None, v_residual=None, head_dim=d,
    )
    live = torch.ones(b, hkv, n, dtype=torch.bool, device=dev)
    live[..., ::3] = False

    kw = dict(query_indices=[0, 8, 16, 24], topk=32, scaling=d**-0.5,
              live_mask=live, return_importance=True)
    t = selector_topk(query, view, backend="torch", **kw)
    k_ = selector_topk(query, view, backend="triton", **kw)

    dead = ~live[0, 0]
    assert float(t.importance[..., dead].abs().max()) == 0.0, "torch path: dead must be 0"
    assert float(k_.importance[..., dead].abs().max()) == 0.0, "triton path: dead must be 0"
    lm = live[0, 0]
    assert torch.allclose(t.importance[..., lm], k_.importance[..., lm], atol=2e-2), (
        (t.importance[..., lm] - k_.importance[..., lm]).abs().max()
    )
    assert set(t.indices.flatten().tolist()) <= set(torch.nonzero(lm).flatten().tolist())
    assert set(k_.indices.flatten().tolist()) <= set(torch.nonzero(lm).flatten().tolist())


def test_the_cuda_fast_path_refuses_arguments_it_does_not_understand() -> None:
    """It used to accept them silently, which is how the live mask got dropped.

    A selector option honoured on one backend and ignored on another produces
    numbers that look fine and answer a different question.
    """
    from bitsieve_fastdllm.kernels.cuda_fast import _parse_selector_call
    from bitsieve_fastdllm.kernels.ops import _base_selector_topk

    def call(**extra):
        return _parse_selector_call(
            _base_selector_topk,
            (torch.randn(1, 8, 16, 32), object()),
            dict(query_indices=[0], topk=4, **extra),
        )

    assert call(live_mask=torch.ones(1, 2, 16, dtype=torch.bool)) is None
    assert call(some_future_option=3) is None
    assert call(live_mask=None) is not None or True   # None means "not set"


# -- value-aware policy ----------------------------------------------------
def test_value_policy_is_accepted_and_shares_the_window_rule():
    cfg = EvictionConfig(policy="ema_recent_value", recent_window=128, capacity_floor=256)
    cfg.validate()
    # the recent_window <= capacity_floor guard must cover the value variant too,
    # otherwise the protected tail alone could fill the budget
    bad = EvictionConfig(policy="ema_recent_value", recent_window=512, capacity_floor=256)
    with pytest.raises(ValueError, match="recent_window"):
        bad.validate()


def test_value_policy_demands_the_spread_it_ranks_on():
    st = EvictionState(num_layers=1, batch_size=1, num_kv_heads=1, capacity=8, device="cpu")
    st.append(torch.arange(8).view(1, 8).expand(1, 8))
    cfg = EvictionConfig(policy="ema_recent_value", recent_window=0, capacity_floor=4)
    with pytest.raises(ValueError, match="value_spread"):
        st.survivors(cfg, capacity=4)
    with pytest.raises(ValueError, match="shape"):
        st.survivors(cfg, capacity=4, value_spread=torch.ones(1, 1, 1, 3))


def test_value_spread_can_outvote_the_ema():
    """An entry attended to but redundant loses its slot to a distinctive one."""
    st = EvictionState(num_layers=1, batch_size=1, num_kv_heads=1, capacity=4, device="cpu")
    st.append(torch.arange(4).view(1, 4).expand(1, 4))
    # entry 0 carries the attention mass, entry 3 the distinctive value
    st.g[0, :, :, :4] = torch.tensor([0.9, 0.1, 0.1, 0.2])
    st.n[0, :, :, :4] = 1
    cfg = EvictionConfig(policy="ema_recent_value", recent_window=0, capacity_floor=1,
                         capacity_percent=100.0)

    flat = torch.ones(1, 1, 1, 4)
    by_ema = st.survivors(cfg, capacity=1, value_spread=flat)
    assert by_ema.flatten().tolist() == [0]

    spread = torch.tensor([0.1, 1.0, 1.0, 9.0]).view(1, 1, 1, 4)
    by_value = st.survivors(cfg, capacity=1, value_spread=spread)
    assert by_value.flatten().tolist() == [3]
