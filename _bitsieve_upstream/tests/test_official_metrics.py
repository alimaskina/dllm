from bitsieve_fastdllm.eval.metrics import score_prediction


def test_longbench_qa_uses_official_normalization_and_f1():
    assert score_prediction("2wikimqa", "The Eiffel Tower", ["eiffel tower"]) == 1.0


def test_repobench_strips_fences_and_uses_fuzzy_ratio():
    assert score_prediction("repobench-p", "```python\nreturn x\n```", ["return x"]) == 1.0


def test_qmsum_returns_official_rouge_l_f_score():
    score = score_prediction("qmsum", "the team met", ["the team met"])
    assert score == 1.0
