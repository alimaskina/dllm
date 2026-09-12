"""Coverage must report absolute retained mass, not only selector optimality.

`mass` is normalised by the best achievable mass at the budget, so it answers
"did this selector pick well?" and NOT "how much attention mass survived?".
Those diverge exactly where it matters: measured on hotpotqa at a fixed
128-entry budget, the all-query selector reports mass 1.0000 while the task
score falls by 0.09. Without the absolutes, a reader sees 1.0000 and concludes
nothing was lost.
"""
import torch

from bitsieve_fastdllm.runtime.trace import CoverageRecord, RunTrace


def test_mass_is_relative_and_mass_abs_is_not():
    # One head, prefix of 4, mass concentrated but with a real tail.
    ref = torch.tensor([[[0.5, 0.3, 0.15, 0.05]]])
    k = 2
    ref_vals, _ = torch.topk(ref, k, dim=-1)
    best_abs = ref_vals.sum(-1)                       # 0.8 - the budget's ceiling
    sel = torch.tensor([[[0, 1]]])                    # the optimal choice
    got_abs = ref.gather(-1, sel).sum(-1)             # 0.8
    assert torch.allclose(got_abs / best_abs, torch.tensor([[1.0]]))
    # Optimal selector, yet a fifth of the attention mass is gone to the budget.
    assert torch.allclose(best_abs, torch.tensor([[0.8]]))


def test_summary_reports_both_and_their_ratio_is_the_relative_mass():
    trace = RunTrace()
    trace.coverage.append(
        CoverageRecord(
            block=0, layer=0, old_cache_len=4, selected_k=2, selector_queries=1,
            mass=[1.0], overlap=[1.0], mass_abs=[0.8], ceiling_abs=[0.8],
        )
    )
    s = trace.coverage_summary()
    assert s["mass_mean"] == 1.0
    assert s["mass_abs_mean"] == 0.8
    assert s["ceiling_abs_mean"] == 0.8
    assert s["mass_abs_mean"] / s["ceiling_abs_mean"] == s["mass_mean"]


def test_summary_omits_the_absolutes_for_traces_written_before_they_existed():
    # Older runs have no mass_abs; reporting 0.0 would read as "kept nothing".
    trace = RunTrace()
    trace.coverage.append(
        CoverageRecord(
            block=0, layer=0, old_cache_len=4, selected_k=2, selector_queries=1,
            mass=[0.9], overlap=[0.5],
        )
    )
    s = trace.coverage_summary()
    assert s["mass_mean"] == 0.9
    assert "mass_abs_mean" not in s
    assert "ceiling_abs_mean" not in s
