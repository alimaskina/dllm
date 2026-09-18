"""An approximation printed above the exact ceiling is a finding, significant or not.

The reviewer reads the table, not the t-statistic: a sparse row sitting above
dense reads as a broken experiment whatever the caption says, and "not
significant" does not un-print it. narrativeqa at n=60 forced this - dense 0.289
against k4v4 0.309 and mage 0.292, nothing close to significant, and the old
significance-gated check called the task `ok`.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "rank", Path(__file__).resolve().parents[1] / "scripts" / "rank_task_orderings.py"
)
rank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rank)

D, M, Q, H = rank.DENSE, rank.MAGE, rank.QUANT, rank.HERALD


def _scores(dense, mage, quant, herald, n=60, jitter=0.01):
    """Per-example scores around each mean, with independent per-arm jitter.

    The jitter has to differ BETWEEN arms or every paired difference is a
    constant, the SE collapses to zero and nothing can be called separated.
    """
    import random

    def arm(mean, seed, jitter):
        rng = random.Random(seed)
        return {str(i): mean + rng.uniform(-jitter, jitter) for i in range(n)}

    return {
        D: arm(dense, 1, jitter), M: arm(mage, 2, jitter),
        Q: arm(quant, 3, jitter), H: arm(herald, 4, jitter),
    }


def test_the_expected_ordering_is_not_flagged():
    # dense on top, herald at the bottom - gov_report's shape.
    rec = rank.analyse("t", _scores(0.340, 0.290, 0.288, 0.269))
    assert rec["verdict"] == "ok", rec["notes"]


def test_a_sparse_arm_above_dense_is_flagged_even_when_far_from_significant():
    # narrativeqa's shape at n=60: per-example QA-F1 spread swamps the gaps,
    # so nothing is significant and the table still shows k4v4 on top.
    rec = rank.analyse("t", _scores(0.289, 0.292, 0.309, 0.275, jitter=0.30))
    assert rec["verdict"] == "inverted"
    joined = " ".join(rec["notes"])
    assert "> dense" in joined
    assert "visible in the table" in joined


def test_the_note_says_which_arms_are_above_dense():
    rec = rank.analyse("t", _scores(0.289, 0.292, 0.309, 0.275, jitter=0.30))
    joined = " ".join(rec["notes"])
    assert M in joined and Q in joined
    # herald is below dense here and must not be named.
    assert f"{H} > dense" not in joined


def test_an_exact_tie_is_not_an_inversion():
    # Identical per-example scores, not merely equal means - the check must key
    # off the paired difference being zero, not off rounding.
    scores = _scores(0.300, 0.300, 0.290, 0.270)
    scores[M] = dict(scores[D])
    rec = rank.analyse("t", scores)
    joined = " ".join(rec["notes"])
    assert f"{M} > dense" not in joined, rec["notes"]
