"""A mean on a discrete metric can be one flipped example; the report must say so.

lsht at n=20 showed `mage-herald` -0.050 and was read as an effect of the wrong
sign. It was a single differing example out of 20, the other 19 tied. trec at
the same n showed +0.250 from 5 differing examples, all 5 the same way. The two
means look like results of the same kind and are not.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "rank", Path(__file__).resolve().parents[1] / "scripts" / "rank_task_orderings.py"
)
rank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rank)


def test_counts_only_examples_that_actually_differ():
    a = {"1": 1.0, "2": 0.0, "3": 1.0, "4": 1.0}
    b = {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}
    assert rank.n_differing(a, b) == (1, 4)


def test_ties_everywhere_report_zero_not_the_sample_size():
    a = {"1": 0.5, "2": 0.5}
    assert rank.n_differing(a, dict(a)) == (0, 2)


def test_only_the_overlap_is_counted():
    # A half-finished arm must not silently shrink or inflate the denominator.
    a = {"1": 1.0, "2": 0.0, "3": 1.0}
    b = {"1": 0.0, "2": 0.0}
    assert rank.n_differing(a, b) == (1, 2)


def test_the_lsht_and_trec_cases_are_distinguishable():
    # lsht: one flip of twenty, mean -0.05.
    lsht_m = {str(i): 0.0 for i in range(20)}
    lsht_h = dict(lsht_m); lsht_h["7"] = 1.0
    # trec: five flips of twenty, all the same way, mean +0.25.
    trec_m = {str(i): 0.0 for i in range(20)}
    trec_h = dict(trec_m)
    for i in range(5):
        trec_m[str(i)] = 1.0
    assert rank.n_differing(lsht_m, lsht_h) == (1, 20)
    assert rank.n_differing(trec_m, trec_h) == (5, 20)
    # Both means are large; only the counts separate a result from a coin flip.
    assert abs(rank.paired_diff(lsht_m, lsht_h)[0]) == 0.05
    assert abs(rank.paired_diff(trec_m, trec_h)[0]) == 0.25


def test_coverage_loader_surfaces_the_budget_ceiling():
    """The relative figure alone reads as "nothing lost" when most of it is.

    On gov_report at a fixed 128-entry budget the all-query selector reports
    mass 1.0000 - it took essentially everything reachable - while only 73% of
    the prefix attention mass is reachable at that budget at all. Reporting the
    ceiling separates the loss the budget forces on every selector from the loss
    a particular selector adds.
    """
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        d = Path(tmp)
        rows = [
            {"variant": "sparse_fp16_all", "pass": "coverage",
             "runtime": {"coverage": {"mass_mean": 0.9999, "overlap_mean": 0.99,
                                      "ceiling_abs_mean": 0.733}}},
            # An older row with no ceiling recorded must not poison the mean.
            {"variant": "sparse_fp16_all", "pass": "coverage",
             "runtime": {"coverage": {"mass_mean": 0.9997, "overlap_mean": 0.98}}},
        ]
        (d / "sparse_fp16_all.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        cov = rank.load_task_coverage(d)
    mass, overlap, n, ceiling = cov["sparse_fp16_all"]
    assert n == 2
    assert abs(mass - 0.9998) < 1e-9
    assert ceiling == 0.733


def test_coverage_ceiling_is_none_for_older_runs():
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "sparse_fp16_all.jsonl").write_text(
            json.dumps({"variant": "sparse_fp16_all", "pass": "coverage",
                        "runtime": {"coverage": {"mass_mean": 0.95,
                                                 "overlap_mean": 0.9}}}) + "\n",
            encoding="utf-8",
        )
        cov = rank.load_task_coverage(d)
    assert cov["sparse_fp16_all"][3] is None


def test_sign_test_counts_wins_and_losses_not_ties():
    a = {"1": 1.0, "2": 0.0, "3": 1.0, "4": 1.0}
    b = {"1": 0.0, "2": 1.0, "3": 1.0, "4": 0.0}
    wins, losses, p = rank.sign_test(a, b)
    assert (wins, losses) == (2, 1)
    assert 0.0 < p <= 1.0


def test_a_clean_sweep_is_significant_and_a_split_is_not():
    # trec's shape: seven disagreements, all one way.
    sweep_a = {str(i): (1.0 if i < 7 else 0.0) for i in range(60)}
    sweep_b = {str(i): 0.0 for i in range(60)}
    wins, losses, p = rank.sign_test(sweep_a, sweep_b)
    assert (wins, losses) == (7, 0)
    assert p == pytest.approx(2 / 2 ** 7, abs=1e-9)   # 0.015625
    # dense-mage's shape on the same task: three disagreements, near-even.
    split_a = {"1": 1.0, "2": 1.0, "3": 0.0}
    split_b = {"1": 0.0, "2": 0.0, "3": 1.0}
    assert rank.sign_test(split_a, split_b) == (2, 1, 1.0)


def test_all_ties_report_no_p_value_rather_than_a_fake_one():
    a = {"1": 0.5, "2": 0.5}
    wins, losses, p = rank.sign_test(a, dict(a))
    assert (wins, losses) == (0, 0)
    assert p != p   # NaN: nothing was compared, so there is no p to report


def test_sign_test_is_robust_where_the_mean_is_not():
    """The mean on trec halved from +0.250 to +0.125 as n went 20 -> 56.

    The sign test's summary - every disagreement went the same way - held
    throughout, which is why it belongs beside the mean.
    """
    small_a = {str(i): (1.0 if i < 5 else 0.0) for i in range(20)}
    small_b = {str(i): 0.0 for i in range(20)}
    big_a = {str(i): (1.0 if i < 7 else 0.0) for i in range(56)}
    big_b = {str(i): 0.0 for i in range(56)}
    assert rank.paired_diff(small_a, small_b)[0] == pytest.approx(0.25)
    assert rank.paired_diff(big_a, big_b)[0] == pytest.approx(0.125)
    assert rank.sign_test(small_a, small_b)[:2] == (5, 0)
    assert rank.sign_test(big_a, big_b)[:2] == (7, 0)
