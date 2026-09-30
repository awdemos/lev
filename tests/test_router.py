import pytest

from lev.router import question_from_dict
from lev.types import ChoiceQuestion, ScoreQuestion


def test_parses_choice_question():
    q = question_from_dict(
        {
            "type": "choice",
            "question": "Which department?",
            "labels": {"billing": "invoices", "other": "rest"},
        }
    )
    assert isinstance(q, ChoiceQuestion)
    assert q.labels == {"billing": "invoices", "other": "rest"}


def test_parses_score_question():
    q = question_from_dict({"type": "score", "question": "How urgent?"})
    assert isinstance(q, ScoreQuestion)


def test_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown type"):
        question_from_dict({"type": "rank"})


def test_rejects_choice_without_labels():
    with pytest.raises(ValueError, match="labels"):
        question_from_dict({"type": "choice", "question": "q?"})


def test_rejects_less_than_two_labels():
    with pytest.raises(ValueError, match="two"):
        question_from_dict(
            {"type": "choice", "question": "q?", "labels": {"only": "one"}}
        )
