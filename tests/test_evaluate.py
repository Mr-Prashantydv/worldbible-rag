from pathlib import Path

import pytest

from worldbible_rag import evaluate
from worldbible_rag.evaluate import Case, first_hit_rank, matches

EVALS = Path(__file__).resolve().parents[1] / "evals"


def test_target_matching():
    assert matches("joy#personality", "joy#personality")
    assert matches("joy#notes-2", "joy#notes")  # split pieces of one section
    assert matches("joy#anything", "joy")  # a bare slug matches the whole record
    assert not matches("joy#personality-traits", "joy#personality")
    assert not matches("joys-lab#overview", "joy")


def test_first_hit_rank():
    assert first_hit_rank(["a#x", "b#y", "c#z"], ["b"]) == 2
    assert first_hit_rank(["a#x"], ["b"]) is None


def test_eval_runs_every_mode_and_reports_metrics(retriever):
    cases = evaluate.load_cases(EVALS / "sample.yaml")
    report = evaluate.run(retriever, cases)
    assert report["cases"] == len(cases)
    for metrics in report["modes"].values():
        assert 0 <= metrics["hit@1"] <= metrics["hit@3"] <= metrics["hit@5"] <= 1
        assert set(metrics["ranks"]) == {c.id for c in cases}
    assert "MRR" in evaluate.format_table(report)


def test_eval_refuses_a_target_that_is_not_in_the_corpus(retriever):
    with pytest.raises(ValueError, match="not found in the corpus"):
        evaluate.run(retriever, [Case(id="typo", question="q", expect=["bolt#no-such-section"])])
