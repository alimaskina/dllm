"""The sweep's arm matrix and its iso-memory grouping.

The whole experiment turns on one comparison: arms costing the SAME bits, split
differently between precision and entry count. If the grouping is wrong the
table still prints and says nothing.
"""
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load("coverage_sweep", "scripts/coverage_sweep.py")


def test_ten_arms_with_the_specified_baselines():
    math_arms = sweep.build_arms("gsm8k")
    lb_arms = sweep.build_arms("hotpotqa")
    assert len(math_arms) == len(lb_arms) == 10
    # Math gets a fixed entry count (its prompts are ~100 tokens, so a percentage
    # collapses to a handful of entries); LongBench gets the percentage.
    assert math_arms[0] == {"name": "fp16", "bits": 16, "topk": 32}
    assert lb_arms[0] == {"name": "fp16", "bits": 16, "topk_percent": 2.5}
    assert {a["bits"] for a in math_arms} == {16, 4, 3, 2}
    assert {a["topk"] for a in math_arms} == {32, 64, 128}
    assert {a["topk_percent"] for a in lb_arms} == {2.5, 5.0, 10.0}


def test_three_bit_arms_are_present_and_unpacked_anywhere_else():
    # 3 bits has no packed kernel; the sweep reaches it only through the
    # round-tripped quantizer, which is the reason this experiment can run at all.
    names = {a["name"] for a in sweep.build_arms("gsm8k")}
    assert {"3bit", "3bit_x2", "3bit_x4"} <= names


@pytest.mark.parametrize("benchmark", ["gsm8k", "hotpotqa"])
def test_the_iso_memory_pairs_are_exactly_equal(benchmark):
    arms = {a["name"]: a for a in sweep.build_arms(benchmark)}
    mem = {n: sweep.relative_memory(a, benchmark) for n, a in arms.items()}
    # These three pairs are the experiment. Equality has to be exact, not close:
    # the report groups by the value, so a float wobble silently splits a pair
    # and the comparison vanishes from the output.
    assert mem["4bit_x4"] == mem["fp16"] == 1.0
    assert mem["2bit_x4"] == mem["4bit_x2"] == 0.5
    assert mem["2bit_x2"] == mem["4bit"] == 0.25


def test_memory_falls_with_bits_at_a_fixed_count():
    arms = {a["name"]: a for a in sweep.build_arms("gsm8k")}
    mem = [sweep.relative_memory(arms[n], "gsm8k") for n in ("fp16", "4bit", "3bit", "2bit")]
    assert mem == sorted(mem, reverse=True)


def test_the_generation_config_uses_the_baseline_arm():
    # Every arm is scored on the baseline arm's trajectory; a run that generated
    # under some other budget would be measuring a different set of queries.
    for benchmark in ("gsm8k", "hotpotqa"):
        arms = sweep.build_arms(benchmark)
        cfg = sweep.build_config(benchmark, arms)
        base = arms[0]
        assert cfg.selector.topk_percent == base.get("topk_percent")
        if "topk" in base:
            assert cfg.selector.topk == base["topk"]
        assert cfg.coverage_diagnostics, "arms need the fp16 reference"
        assert len(cfg.coverage_arms) == 10
