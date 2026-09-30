from lev import Calibration, Decision, decide
from lev.types import ChoiceQuestion

QUESTION = ChoiceQuestion(
    question="Which department should handle this?",
    labels={
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else",
    },
)

EASY_CALIBRATION = Calibration(a=0.5, b=0.0)


def test_auto_path_when_calibrated_and_above_threshold():
    d = decide("please refund my duplicate invoice payment", QUESTION,
               calibration=EASY_CALIBRATION, threshold=0.5)
    assert d.action == "auto"
    assert d.label == "billing"
    assert d.shortlist == ()
    assert d.calibrated is True
    assert d.confidence >= 0.5


def test_uncalibrated_always_escalates_even_at_high_confidence():
    d = decide("please refund my duplicate invoice payment", QUESTION)
    raw = decide("please refund my duplicate invoice payment", QUESTION,
                 calibration=EASY_CALIBRATION)
    assert raw.confidence > 0.9  # raw confidence really is high
    assert d.action == "escalate"
    assert d.label is None
    assert d.calibrated is False
    assert len(d.shortlist) == 2


def test_calibrated_below_threshold_escalates():
    hard_calib = Calibration(a=1.0, b=0.0)
    d = decide("invoice bug", QUESTION, calibration=hard_calib, threshold=0.99)
    assert d.action == "escalate"
    assert len(d.shortlist) == 2


def test_shortlist_ordered_by_probability_desc():
    d = decide("invoice bug", QUESTION)
    probs = [p for _, p in d.shortlist]
    assert probs == sorted(probs, reverse=True)
    labels = [label for label, _ in d.shortlist]
    dist = decide("invoice bug", QUESTION).shortlist
    assert labels == [l for l, _ in dist]


def test_decision_dataclass_defaults():
    d = Decision(action="escalate", label=None, shortlist=(),
                 confidence=0.5, calibrated=False, margin=0.0, iterations=1)
    assert d.action == "escalate"


def test_decide_rejects_score_questions():
    from lev.types import ScoreQuestion
    import pytest

    with pytest.raises(ValueError, match="only supports choice questions"):
        decide("urgent outage", ScoreQuestion(question="how urgent?"))
