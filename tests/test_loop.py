import pytest

from lev.loop import classify
from lev.types import ChoiceQuestion, ScoreQuestion

QUESTION = ChoiceQuestion(
    question="Which department should handle this?",
    labels={
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else",
    },
)


def test_obvious_input_converges_in_one_pass():
    result = classify(
        "we were billed twice for March, please refund the duplicate payment",
        QUESTION,
    )
    assert result.label == "billing"
    assert result.iterations == 1
    assert result.confidence >= 0.9


def test_ambiguous_input_spends_more_iterations_or_budget():
    result = classify(
        "billed twice but a system error caused the duplicate invoice",
        QUESTION,
        threshold=0.99,
        max_iters=6,
    )
    # Empirically the loop needs 2 passes to cross 0.99, and the final
    # distribution is sharpened (top label > 0.99).
    assert result.iterations >= 2
    top = max(result.distribution.values())
    assert top > 0.99
    assert sum(result.distribution.values()) == pytest.approx(1.0)


def test_iteration_budget_is_respected():
    result = classify(
        "billed twice but the system error caused the duplicate invoice",
        QUESTION,
        threshold=0.999999,
        max_iters=3,
    )
    assert result.iterations <= 3


def test_zero_evidence_state_returns_uniform_distribution():
    result = classify("zzz qqq vvv", QUESTION)
    for p in result.distribution.values():
        assert p == pytest.approx(1 / 3)


def test_score_question_returns_int_in_range():
    q = ScoreQuestion(question="How urgent is this?")
    result = classify("URGENT immediately critical now now now", q)
    assert 0 <= result.score <= 100
    assert isinstance(result.score, int)
    assert result.confidence > 0.3
