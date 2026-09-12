"""The suite must be able to run LongBench on a fixed KV budget, not just a percentage.

A percent-of-prefix budget stops discriminating between selectors exactly where
the method is supposed to matter: 5% of a 20k prefix is 1000 entries, enough for
any selector to reproduce nearly all the attention mass. The KV-cache literature
fixes the budget (128/256/512) regardless of prefix length for that reason.
"""
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_suite", _ROOT / "scripts" / "run_suite.py")
run_suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_suite)

SPARSE_STEMS = [stem for stem, has_selector in run_suite.VARIANTS.values() if has_selector]


def _cfg(stem, benchmark, **kw):
    kw.setdefault("topk_pct", 5.0)
    kw.setdefault("math_topk", 64)
    kw.setdefault("coverage", False)
    return run_suite.load_config(stem, benchmark=benchmark, **kw)


@pytest.mark.parametrize("stem", SPARSE_STEMS)
def test_percent_budget_is_the_default_for_longbench(stem):
    cfg = _cfg(stem, "hotpotqa")
    assert cfg.selector.topk_percent == 5.0


@pytest.mark.parametrize("stem", SPARSE_STEMS)
def test_fixed_budget_replaces_the_percentage_entirely(stem):
    # Both must not be set: a leftover topk_percent would silently win or
    # make the effective budget depend on which the selector reads first.
    cfg = _cfg(stem, "hotpotqa", longbench_topk=128)
    assert cfg.selector.topk_percent is None
    assert cfg.selector.topk == 128


@pytest.mark.parametrize("stem", SPARSE_STEMS)
def test_math_keeps_its_own_fixed_budget_regardless(stem):
    # GSM8K prompts are ~100-300 tokens; it has its own budget and must not be
    # dragged onto the LongBench one.
    cfg = _cfg(stem, "gsm8k", longbench_topk=128, math_topk=64)
    assert cfg.selector.topk_percent is None
    assert cfg.selector.topk == 64


def test_dense_has_no_selector_budget_to_override():
    for kw in ({}, {"longbench_topk": 128}):
        cfg = _cfg("dense_bf16", "hotpotqa", **kw)
        assert cfg.semantic == "dense"
        assert cfg.selector.topk_percent is None


def test_budget_mode_is_recorded_in_the_manifest_settings():
    # run_suite's manifest guard must reject resuming a percent-budget run with
    # a fixed-budget one; that only works if the flag is in the manifest.
    source = (_ROOT / "scripts" / "run_suite.py").read_text(encoding="utf-8")
    assert '"longbench_topk": args.longbench_topk,' in source
