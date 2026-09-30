from collections import Counter

import pytest

from lev.engine import blend, confidence, softmax, tokenize
from lev.types import ChoiceQuestion, ChoiceResult, ScoreQuestion, ScoreResult


def test_tokenize_lowercases_and_counts():
    assert tokenize("Refund! refund the REFUND") == Counter({"refund": 3})


def test_softmax_is_a_distribution():
    dist = softmax({"a": 2.0, "b": 1.0, "c": 0.0})
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["a"] > dist["b"] > dist["c"]


def test_confidence_is_max_probability():
    assert confidence({"a": 0.7, "b": 0.3}) == 0.7


def test_blend_without_prior_softmaxes_raw_scores():
    raw = {"a": 2.0, "b": 0.0}
    dist = blend(None, raw)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["a"] > 0.5


def test_blend_with_prior_sharpens_toward_prior():
    raw = {"a": 2.0, "b": 0.0}
    prior = {"a": 0.9, "b": 0.1}
    dist = blend(prior, raw, alpha=0.5)
    flat = blend(None, raw)
    assert dist["a"] > flat["a"]


def test_question_dataclasses():
    q = ChoiceQuestion(question="q?", labels={"a": "words", "b": "other"})
    assert q.labels["a"] == "words"
    s = ScoreQuestion(question="how urgent?")
    assert isinstance(s, ScoreQuestion)
    r = ChoiceResult(label="a", distribution={"a": 1.0}, confidence=1.0, iterations=1)
    assert r.iterations == 1
    sr = ScoreResult(score=80, confidence=0.6, iterations=2)
    assert sr.score == 80
