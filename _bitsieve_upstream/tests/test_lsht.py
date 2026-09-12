"""lsht is wired in as the Chinese twin of trec.

trec is the only task whose metric separates the selectors, and the reason is
mechanical: its label set is defined by few-shot demonstrations spread through
the prompt, so a single-middle-query selector answers with the right meaning in
the wrong vocabulary ("Golf course" where the closed label set says "Other
location"). lsht has the same shape - 24 closed classes, few-shot, exact-match
scoring - so it tests that explanation on an independent task instead of drawing
twice from trec.

It is also the only Chinese LongBench task that can be run at full official
fidelity here: its metric is classification_score, plain substring matching with
no jieba segmentation anywhere in it.
"""
import pytest

from bitsieve_fastdllm.eval.benchmarks import (
    LONG_BENCH_CONFIGS,
    load_longbench,
    max_new_tokens_for,
    uses_chat_template,
    _longbench_prompt,
)
from bitsieve_fastdllm.eval.metrics import score_prediction


def test_lsht_follows_the_same_protocol_as_trec():
    assert LONG_BENCH_CONFIGS["lsht"] == "lsht"
    assert max_new_tokens_for("lsht") == 64            # dataset2maxlen.json
    assert not uses_chat_template("lsht")              # pred.py's exemption list


def test_lsht_prompt_is_the_official_template():
    prompt = _longbench_prompt("lsht", "CTX", "IN")
    assert prompt == "请判断给定新闻的类别，下面是一些例子。\n\nCTX\nIN"


def test_lsht_scores_through_classification_with_first_line_only():
    classes = ["科学技术", "体育", "军事"]
    assert score_prediction("lsht", "科学技术", ["科学技术"], all_classes=classes) == 1.0
    assert score_prediction("lsht", "体育", ["科学技术"], all_classes=classes) == 0.0
    # Upstream keeps only the first line for lsht, so a conversational preamble
    # discards the answer - which is exactly why the chat template is skipped.
    assert score_prediction(
        "lsht", "这条新闻的类别是：\n科学技术", ["科学技术"], all_classes=classes
    ) == 0.0


def test_lsht_needs_all_classes_like_trec():
    # Without the class list there is no way to run the official algorithm, and
    # guessing a substring match would silently invent a score.
    assert score_prediction("lsht", "科学技术", ["科学技术"], all_classes=None) == 0.0


def test_lsht_examples_load_with_their_class_list():
    examples = load_longbench("lsht", 3)
    assert len(examples) == 3
    for e in examples:
        assert e.metadata["all_classes"], "classification_score cannot run without them"
        assert e.references
        assert e.prompt.startswith("请判断给定新闻的类别")
        # Ends mid-pattern ("类别:"), the signature of a completion-style prompt.
        assert e.prompt.rstrip().endswith("类别：")


def test_the_other_chinese_tasks_stay_unwired():
    # They need jieba-segmented metrics this project does not depend on; wiring
    # them without that would score them wrong while looking official.
    for task in ("dureader", "vcsum", "multifieldqa_zh", "passage_retrieval_zh"):
        assert task not in LONG_BENCH_CONFIGS
