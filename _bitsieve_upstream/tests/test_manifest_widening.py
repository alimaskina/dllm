"""The suite manifest must allow deepening a run and nothing else.

Rerunning the same tasks with a larger --longbench-n to tighten a CI is the
one manifest difference that leaves every finished row valid: the loader takes
a deterministic prefix of the split, so the old ids are a prefix of the new
ones. Every other difference means the finished rows describe something else.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_suite", Path(__file__).resolve().parents[1] / "scripts" / "run_suite.py"
)
run_suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_suite)
err = run_suite._manifest_extension_error


def _manifest(*, n=20, ids=None, topk_pct=5.0, model="m", revision=None):
    ids = ids if ids is not None else [f"id{i}" for i in range(n)]
    return {
        "schema_version": 1,
        "model": {"id": model, "revision": revision, "dtype": "bf16"},
        "datasets": {
            "qasper": {
                "repo": "zai-org/LongBench",
                "revision": None,
                "split": "test",
                "num_examples": len(ids),
                "example_ids": ids,
                "fingerprint": "f" + str(len(ids)),
            }
        },
        "settings": {
            "gsm8k_n": 0,
            "longbench_n": n,
            "longbench_tasks": ["qasper"],
            "variants": ["dense"],
            "topk_pct": topk_pct,
            "longbench_topk": None,
            "math_topk": 64,
            "coverage_pass": False,
        },
    }


def test_identical_manifest_is_accepted():
    assert err(_manifest(), _manifest()) is None


def test_widening_example_count_is_accepted():
    # The whole point: n=20 -> n=60 over the same deterministic prefix.
    assert err(_manifest(n=20), _manifest(n=60)) is None


def test_shrinking_example_count_is_rejected():
    # The finished rows cover 60 examples; a 20-example manifest would silently
    # relabel the run as smaller than the data on disk.
    assert "shrank" in err(_manifest(n=60), _manifest(n=20))


def test_reslicing_the_dataset_is_rejected():
    # Same count, different examples - a shuffled or filtered split. The old
    # ids are no longer a prefix, so the finished rows are not comparable.
    resliced = _manifest(ids=[f"other{i}" for i in range(20)])
    assert "re-sliced" in err(_manifest(n=20), resliced)


def test_widening_with_a_changed_prefix_is_rejected():
    # More examples, but the first 20 are not the old 20.
    resliced = _manifest(ids=[f"other{i}" for i in range(60)])
    assert "re-sliced" in err(_manifest(n=20), resliced)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"topk_pct": 1.0}, "settings differ"),
        ({"model": "other"}, "model differs"),
        ({"revision": "abc"}, "model differs"),
    ],
)
def test_a_different_run_is_still_rejected(kwargs, needle):
    assert needle in err(_manifest(), _manifest(**kwargs))


def test_dropping_a_dataset_is_rejected():
    # In practice settings.longbench_tasks changes with it and reports first,
    # so assert the rejection rather than which check caught it.
    current = _manifest()
    current["datasets"] = {}
    current["settings"]["longbench_tasks"] = []
    assert err(_manifest(), current) is not None


def test_dropping_a_dataset_is_rejected_even_if_settings_still_list_it():
    # Guards the datasets check on its own: a manifest whose settings still
    # name the task but whose dataset block is gone must not slip through.
    current = _manifest()
    current["datasets"] = {}
    assert "dropped" in err(_manifest(), current)


def test_changed_dataset_revision_is_rejected():
    current = _manifest()
    current["datasets"]["qasper"]["revision"] = "v2"
    assert "revision differs" in err(_manifest(), current)


def test_a_setting_added_since_the_earlier_run_is_not_a_difference():
    """A manifest written before a flag existed must still be extendable.

    --longbench-topk was added to the manifest after several runs had been
    written. Their manifests have no such key; new ones carry it as None. Dict
    equality called that a different run and then printed an EMPTY list of
    differences - because the diff itself, using .get(), correctly saw none.
    Two tasks failed mid-sweep on this.
    """
    previous = _manifest()
    del previous["settings"]["longbench_topk"]
    current = _manifest()
    assert current["settings"]["longbench_topk"] is None
    assert err(previous, current) is None
    # ... and widening on top of it still works.
    assert err(previous, _manifest(n=60)) is None


def test_a_setting_added_with_a_REAL_value_is_still_a_difference():
    # The flag being absent before and set now means the budget actually
    # changed; that must still be refused.
    previous = _manifest()
    del previous["settings"]["longbench_topk"]
    current = _manifest()
    current["settings"]["longbench_topk"] = 128
    assert "settings differ" in err(previous, current)
