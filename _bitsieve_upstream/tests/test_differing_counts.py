"""A mean on a discrete metric can be one flipped example; the report must say so.

lsht at n=20 showed `mage-herald` -0.050 and was read as an effect of the wrong
sign. It was a single differing example out of 20, the other 19 tied. trec at
the same n showed +0.250 from 5 differing examples, all 5 the same way. The two
means look like results of the same kind and are not.
"""
import importlib.util
from pathlib import Path

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
