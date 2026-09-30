from collections import Counter

import pytest

from lev.engine import tokenize
from lev.loop import classify
from lev.signals import CharNgramSignal, NegationSignal, SaturationSignal, TokenOverlapSignal
from lev.types import ChoiceQuestion

QUESTION = ChoiceQuestion(
    question="Which department should handle this?",
    labels={
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else",
    },
)


def test_char_ngram_jaccard_known_pair():
    sig = CharNgramSignal()
    # "invoice" and "invoiice" share most 3-grams; disjoint strings share none.
    q = ChoiceQuestion(question="q?", labels={"a": "invoice", "b": "quantum"})
    scores = sig.score("invoiice refund", Counter(), q)
    assert scores["a"] > 0.2
    assert scores["b"] == 0.0


def test_char_ngram_empty_union_is_zero():
    sig = CharNgramSignal()
    q = ChoiceQuestion(question="q?", labels={"a": "", "b": "x"})
    scores = sig.score("", Counter(), q)
    assert scores["a"] == 0.0
    assert scores["b"] == 0.0


def test_saturation_full_when_all_evidence_matched():
    sig = SaturationSignal()
    q = ChoiceQuestion(question="q?", labels={"billing": "invoices, payments, refunds"})
    tokens = tokenize("billing invoice payment refund")
    assert sig.score("", tokens, q)["billing"] == 1.0


def test_saturation_fractional_when_partially_matched():
    sig = SaturationSignal()
    q = ChoiceQuestion(question="q?", labels={"billing": "invoices, payments, refunds"})
    tokens = tokenize("invoice")
    assert sig.score("", tokens, q)["billing"] == pytest.approx(0.25)


def test_negation_penalizes_label():
    sig = NegationSignal()
    neg_text = "not a billing refund, this is a technical outage"
    unneg_text = "not a technical outage, this is a billing refund"
    negated = sig.score(neg_text, tokenize(neg_text), QUESTION)
    unnegated = sig.score(unneg_text, tokenize(unneg_text), QUESTION)
    assert negated["billing"] == 0.0
    assert unnegated["billing"] > 0.0
    assert negated["technical"] > negated["billing"]


def test_negation_is_neutral_without_negations():
    sig = NegationSignal()
    scores = sig.score("please refund my invoice", Counter(), QUESTION)
    assert scores == {label: 0.0 for label in QUESTION.labels}


def test_token_overlap_matches_engine_raw_scores():
    sig = TokenOverlapSignal()
    tokens = tokenize("refund the invoice please")
    expected = {label: 3.0 * n for label, n in
                [("billing", 2), ("technical", 0), ("other", 0)]}
    assert sig.score("", tokens, QUESTION) == expected


def test_typo_input_routed_by_ngram_pass():
    result = classify("refnd my invoiice", QUESTION)
    assert result.label == "billing"
    assert result.iterations >= 2
