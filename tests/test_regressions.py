import json
import subprocess
import sys
from collections import Counter

import pytest

from lev.engine import tokenize
from lev.loop import classify
from lev.types import ChoiceQuestion, ScoreQuestion

CHOICE = ChoiceQuestion(question="q?", labels={"a": "words", "b": "other"})


def test_classify_rejects_empty_labels():
    with pytest.raises(ValueError, match="labels must be non-empty"):
        classify("hello", ChoiceQuestion(question="q?", labels={}))


def test_score_balanced_input_reports_true_confidence_and_budget():
    result = classify("urgent fine", ScoreQuestion(question="how urgent?"),
                      threshold=0.9, max_iters=4)
    assert result.confidence < 0.9
    assert result.iterations == 4


def test_broken_counts_as_urgent():
    result = classify("broken", ScoreQuestion(question="how urgent?"))
    assert result.score > 50


def test_stemmer_preserves_ss_us_is_and_refunds():
    assert tokenize("class series refunds") == Counter(
        {"class": 1, "series": 1, "refund": 1}
    )


def test_classify_validates_threshold_and_max_iters():
    with pytest.raises(ValueError):
        classify("hello", CHOICE, threshold=1.5)
    with pytest.raises(ValueError):
        classify("hello", CHOICE, max_iters=0)


def test_cli_rejects_non_list_questions(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": "abc"}))
    proc = subprocess.run(
        [sys.executable, "-m", "lev.cli", "classify", "hello",
         "--questions", str(questions)],
        capture_output=True, text=True, cwd=".",
    )
    assert proc.returncode == 1
    assert "lev: error:" in proc.stderr
