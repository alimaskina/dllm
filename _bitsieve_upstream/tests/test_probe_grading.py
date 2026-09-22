"""The benchmark probe's graders, pinned on real generations.

Every multiple-choice case below is a verbatim tail from a Fast-dLLM-v2 run on
MMLU-Pro. They exist because the first version of the fallback searched for the
*first* "answer|option" in the text and so read the model's enumeration of the
choices instead of its verdict -- which silently turned correct answers into
misses and moved the reported score by 0.20 on ten examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from probe_benchmarks import boxed, grade_math, grade_mc  # noqa: E402


def test_boxed_matches_nested_braces():
    assert boxed(r"so \boxed{\frac{1}{2}} follows") == r"\frac{1}{2}"
    assert boxed("no box here") is None
    assert boxed(r"empty \boxed{}") == ""


def test_mc_prefers_the_verdict_over_the_enumeration():
    # The model walks the options ("Option A. ...") and then concludes with I.
    tail = "Option A. foo is wrong. Therefore, the correct option is I. The final answer is I."
    score, letter, rule = grade_mc(tail, "I", 10)
    assert (score, letter) == (1.0, "I"), f"read {letter!r} by {rule}"


def test_mc_reads_the_last_statement_not_the_first():
    tail = "The answer is A at first glance. On reflection, the final answer is J."
    assert grade_mc(tail, "J", 10)[0] == 1.0
    assert grade_mc(tail, "A", 10)[0] == 0.0


def test_mc_scores_a_genuine_miss_as_a_miss():
    tail = "Therefore, the final answer is D. Stakeholders, Diligence, Care and Skill."
    assert grade_mc(tail, "F", 10)[0] == 0.0


def test_mc_boxed_wins_over_any_fallback():
    tail = r"Option A is tempting. \boxed{C}"
    score, letter, rule = grade_mc(tail, "C", 10)
    assert (score, letter, rule) == (1.0, "C", "boxed")


def test_mc_empty_box_falls_through_to_the_text():
    # Seen in the wild: the model names the option and then emits \boxed{}.
    tail = r"the correct answer is option A: Interest. Therefore, the final answer is \boxed{}."
    assert grade_mc(tail, "A", 10)[0] == 1.0


def test_mc_letters_beyond_the_option_count_are_not_candidates():
    # GPQA is A-D; a stray "I" in prose must not be read as an answer.
    assert grade_mc("I think the final answer is B.", "B", 4)[0] == 1.0


def test_math_scores_a_boxed_integer():
    assert grade_math(r"hence \boxed{204}", "204")[0] == 1.0
    assert grade_math(r"hence \boxed{204}", "205")[0] == 0.0


def test_math_ignores_leading_zeros_and_latex_padding():
    assert grade_math(r"\boxed{ 073 }", "73")[0] == 1.0
    assert grade_math(r"\boxed{\frac{1}{2}}", "\\frac{1}{2}")[0] == 1.0


def test_math_returns_a_score_not_the_gold_string():
    # Regression: `nb.lstrip("0") and nb == ng` evaluates to a string, and the
    # old code called float() on it -- crashing on "" and returning 204.0 on "204".
    for pred, gold in [("no digits at all", "5"), (r"\boxed{204}", "204")]:
        score, _, _ = grade_math(pred, gold)
        assert score in (0.0, 1.0)


def test_math_falls_back_to_the_last_number_without_a_box():
    score, extracted, rule = grade_math("after simplifying we get 17", "17")
    assert (score, extracted, rule) == (1.0, "17", "last-number")
