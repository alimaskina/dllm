"""LongBench-E and LongBench v2 must reuse the v1 rules, not re-implement them.

LongBench-E is the same 13 tasks re-bucketed by context length; upstream's
`pred.py` and `eval.py` key every per-task rule (prompt template, generation
budget, chat-template exemption, metric) off the base name with the `_e`
suffix stripped. Getting that wrong silently scores an `_e` task under a
different protocol than its v1 twin and makes the two incomparable.
"""
import pytest

from bitsieve_fastdllm.eval.benchmarks import (
    LONGBENCH_E_BUCKETS,
    LONGBENCH_E_TASKS,
    base_task,
    length_bucket,
    max_new_tokens_for,
    uses_chat_template,
    _longbench_prompt,
)
from bitsieve_fastdllm.eval.metrics import longbench_v2_accuracy, score_prediction


# LONGBENCH_E_TASKS holds the *base* names; the wired benchmark keys carry "_e".
@pytest.mark.parametrize("base", sorted(LONGBENCH_E_TASKS))
def test_every_e_task_inherits_its_base_task_protocol(base):
    task = f"{base}_e"
    assert base_task(task) == base
    assert _longbench_prompt(task, "ctx", "q") == _longbench_prompt(base, "ctx", "q")
    assert max_new_tokens_for(task) == max_new_tokens_for(base)
    assert uses_chat_template(task) == uses_chat_template(base)


def test_base_task_leaves_v1_names_alone():
    # "narrativeqa" must not lose a trailing letter, and neither must a task
    # whose own name ends in e.
    for task in ("narrativeqa", "qasper", "trec", "lcc", "gov_report"):
        assert base_task(task) == task


def test_e_tasks_score_through_the_base_metric():
    # trec is classification + first-line-only; trec_e must behave identically.
    classes = ["Other location", "Animal"]
    pred = "Other location\nbecause it names a place."
    assert score_prediction("trec_e", pred, ["Other location"], all_classes=classes) == 1.0
    assert score_prediction("hotpotqa_e", "The Eiffel Tower", ["eiffel tower"]) == 1.0


def test_few_shot_exemption_survives_the_e_suffix():
    # The bug this guards: trec_e wrapped in a chat template scores ~0.22
    # instead of ~0.6 because the answer lands on the discarded second line.
    assert not uses_chat_template("trec_e")
    assert uses_chat_template("hotpotqa_e")


def test_length_buckets_are_contiguous_and_match_upstream():
    assert [b[2] for b in LONGBENCH_E_BUCKETS] == ["0-4k", "4-8k", "8k+"]
    for (_, prev_hi, _), (lo, _, _) in zip(LONGBENCH_E_BUCKETS, LONGBENCH_E_BUCKETS[1:]):
        assert lo == prev_hi, "a gap or overlap would drop or double-count examples"
    assert length_bucket(0) == "0-4k"
    assert length_bucket(3999) == "0-4k"
    assert length_bucket(4000) == "4-8k"   # boundary belongs to the upper bucket
    assert length_bucket(7999) == "4-8k"
    assert length_bucket(8000) == "8k+"
    assert length_bucket(10**6) == "8k+"


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ("The correct answer is (B)", 1.0),
        ("The correct answer is B", 1.0),
        ("The correct answer is (C)", 0.0),
        # No parseable letter scores 0 rather than crashing or silently
        # passing - a refusal or a truncated generation is a wrong answer.
        ("I am not sure.", 0.0),
        ("", 0.0),
    ],
)
def test_longbench_v2_accuracy_uses_the_official_extraction(prediction, expected):
    assert longbench_v2_accuracy(prediction, "B") == expected


def test_longbench_v2_budget_is_not_the_summarization_default():
    assert max_new_tokens_for("longbench_v2") == 128


def test_e_tasks_are_sampled_across_length_buckets_not_prefix():
    """The _e files are bucket-major, so a raw prefix is all short contexts.

    This is the bug the interleave exists for: before it, the first 24 rows of
    hotpotqa_e were 24/24 in the 0-4k bucket and the length stratification that
    is the entire point of LongBench-E never reached the model.
    """
    from collections import Counter

    from bitsieve_fastdllm.eval.benchmarks import load_longbench

    for task in ("hotpotqa_e", "trec_e"):
        counts = Counter(
            length_bucket(e.metadata["length"]) for e in load_longbench(task, 24)
        )
        assert counts == {"0-4k": 8, "4-8k": 8, "8k+": 8}, (task, counts)


def test_e_task_order_is_prefix_stable_so_a_run_can_be_deepened():
    # run_suite.py's manifest guard allows raising --longbench-n only when the
    # old example ids are a prefix of the new ones.
    from bitsieve_fastdllm.eval.benchmarks import load_longbench

    short = [e.example_id for e in load_longbench("hotpotqa_e", 24)]
    long = [e.example_id for e in load_longbench("hotpotqa_e", 60)]
    assert long[: len(short)] == short


def test_interleaving_does_not_touch_v1_tasks():
    # v1 files are not bucket-major and upstream consumes them in file order.
    from bitsieve_fastdllm.eval.benchmarks import load_longbench

    ids = [e.example_id for e in load_longbench("hotpotqa", 5)]
    assert ids == [e.example_id for e in load_longbench("hotpotqa", 20)][:5]


def test_rows_with_no_usable_length_are_kept_last_not_dropped():
    from bitsieve_fastdllm.eval.benchmarks import _interleave_length_buckets

    rows = [
        {"_id": "short", "length": 100},
        {"_id": "nolen"},
        {"_id": "mid", "length": 5000},
        {"_id": "bad", "length": "8000"},
        {"_id": "long", "length": 20000},
    ]
    out = [r["_id"] for r in _interleave_length_buckets(rows)]
    assert out[:3] == ["short", "mid", "long"]
    assert sorted(out[3:]) == ["bad", "nolen"]
