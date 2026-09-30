import pytest

import lev.loop
from lev.loop import classify
from lev.types import ChoiceQuestion

QUESTION = ChoiceQuestion(
    question="Which department should handle this?",
    labels={
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
    },
)


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        classify("refund my invoice", QUESTION, backend="gpt-5")


def test_signals_and_backend_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        classify("refund my invoice", QUESTION, signals=(), backend="heuristic")


def test_auto_falls_back_to_heuristic_when_onnx_unavailable(monkeypatch):
    monkeypatch.setattr(lev.loop, "_onnx_available", lambda: False)
    auto = classify("please refund my invoice", QUESTION, backend="auto")
    heuristic = classify("please refund my invoice", QUESTION, backend="heuristic")
    assert auto.label == heuristic.label
    assert auto.distribution == heuristic.distribution


def test_auto_uses_onnx_chain_when_available(monkeypatch):
    monkeypatch.setattr(lev.loop, "_onnx_available", lambda: True)
    seen = []

    class FakeSignal:
        name = "fake_onnx"

        def score(self, text, tokens, question):
            seen.append(text)
            return {label: 1.0 for label in question.labels}

    monkeypatch.setattr(lev.loop, "_ONNX_CHAIN", (FakeSignal(),))
    result = classify("please refund my invoice", QUESTION, backend="auto")
    assert seen, "auto with onnx available must run the onnx chain"
    assert result.label == "billing"
