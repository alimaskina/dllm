import pytest

from bitsieve_fastdllm.eval.benchmarks import (
    LONG_BENCH_CONFIGS,
    _longbench_prompt,
    uses_chat_template,
)
from bitsieve_fastdllm.eval.metrics import score_prediction


def test_longbench_qa_uses_official_normalization_and_f1():
    assert score_prediction("2wikimqa", "The Eiffel Tower", ["eiffel tower"]) == 1.0


def test_repobench_strips_fences_and_uses_fuzzy_ratio():
    assert score_prediction("repobench-p", "```python\nreturn x\n```", ["return x"]) == 1.0


def test_qmsum_returns_official_rouge_l_f_score():
    # The official `rouge` package divides by (p + r + 1e-8), so a perfect match scores
    # 0.999999995, never exactly 1.0 - asserting == 1.0 here fails against the very
    # implementation this test exists to pin.
    score = score_prediction("qmsum", "the team met", ["the team met"])
    assert score == pytest.approx(1.0, abs=1e-6)


def test_narrativeqa_is_qa_f1_not_rouge():
    # A prior version of this dispatch routed narrativeqa to ROUGE-L, which is not the
    # official LongBench metric for it (qa_f1_score) - lexical-overlap F1 and ROUGE-L
    # disagree on this pair, so this would catch a regression back to the wrong family.
    # Normalised: pred -> ["guest"], ref -> ["he","is","guest","in","home","of","mulvilles"].
    # One token in common, so precision 1/1 and recall 1/7, giving F1 = 2/8 = 0.25.
    pred, ref = "a guest", "He is a guest in the home of the Mulvilles."
    assert score_prediction("narrativeqa", pred, [ref]) == pytest.approx(0.25, abs=1e-6)


def test_gov_report_and_multi_news_and_samsum_use_rouge():
    pred, ref = "The team discussed the budget for next quarter.", "The team discussed the budget."
    for task in ("gov_report", "multi_news", "samsum"):
        score = score_prediction(task, pred, [ref])
        assert 0.0 < score < 1.0


def test_trec_classification_needs_all_classes_and_first_line_only():
    all_classes = ["Other location", "Animal", "City"]
    pred = "Other location\nBecause the passage is about a place, not a person or a date."
    assert score_prediction("trec", pred, ["Other location"], all_classes=all_classes) == 1.0
    # Without all_classes there is no way to disambiguate a substring match from the
    # official algorithm's normalisation - it must not silently return a nonzero score.
    assert score_prediction("trec", pred, ["Other location"], all_classes=None) == 0.0


def test_passage_count_and_passage_retrieval_en():
    assert score_prediction("passage_count", "there are 8 unique paragraphs", ["8"]) == 1.0
    assert score_prediction(
        "passage_retrieval_en", "The answer is Paragraph 15.", ["Paragraph 15"]
    ) == 1.0


def test_every_wired_longbench_task_has_an_official_prompt_template():
    for task in set(LONG_BENCH_CONFIGS.values()):
        # Raises KeyError if a task is wired into LONG_BENCH_CONFIGS without a matching
        # entry in _LONGBENCH_OFFICIAL_PROMPTS - see that dict's docstring.
        assert _longbench_prompt(task, "context", "question")


def test_few_shot_tasks_skip_the_chat_template():
    # THUDM/LongBench's pred.py: `if dataset not in [...]: prompt = build_chat(...)`.
    # These prompts end mid-pattern and the model is meant to continue them; wrapped in
    # a chat turn the model answers conversationally and the scorer's first-line-only
    # rule then discards the answer.
    for task in ("trec", "triviaqa", "samsum", "lcc", "repobench-p"):
        assert not uses_chat_template(task), task
    for task in ("narrativeqa", "qasper", "hotpotqa", "2wikimqa", "gov_report", "qmsum"):
        assert uses_chat_template(task), task
