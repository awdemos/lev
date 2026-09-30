"""Multi-signal scorers for the choice-classification loop.

Each signal scores the state against every label independently; the loop
blends one signal's raw scores per pass with the prior distribution.
"""
import re
from collections import Counter
from typing import Protocol

from lev.engine import EVIDENCE_WEIGHT, raw_choice_scores, tokenize
from lev.types import ChoiceQuestion

_WORD_RE = re.compile(r"[a-z0-9']+")
_NEGATIONS = ("not", "no", "never", "without", "isn't", "aren't", "wasn't",
              "weren't", "don't", "doesn't", "didn't", "won't", "can't",
              "couldn't", "shouldn't", "wouldn't")
_NEGATION_WINDOW = 5


class Signal(Protocol):
    """A deterministic scorer: raw logit-ish scores per label."""

    name: str

    def score(
        self, text: str, tokens: Counter[str], question: ChoiceQuestion
    ) -> dict[str, float]: ...


class TokenOverlapSignal:
    """Presence-based evidence scoring (the v1 signal)."""

    name = "token_overlap"

    def score(
        self, text: str, tokens: Counter[str], question: ChoiceQuestion
    ) -> dict[str, float]:
        return raw_choice_scores(tokens, question.labels)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    chars = "".join(c if (c.isalnum() or c == " ") else " " for c in text.lower())
    chars = f" {chars.strip()} "
    return {chars[i:i + n] for i in range(len(chars) - n + 1)} if len(chars) >= n else set()


class CharNgramSignal:
    """Character 3-gram Jaccard between the state and each label's
    `label + criteria` text; catches typos and morphology the stemmer misses."""

    name = "char_ngram"

    def score(
        self, text: str, tokens: Counter[str], question: ChoiceQuestion
    ) -> dict[str, float]:
        state = _char_ngrams(text)
        scores: dict[str, float] = {}
        for label, criteria in question.labels.items():
            target = _char_ngrams(f"{label} {criteria}")
            union = state | target
            scores[label] = len(state & target) / len(union) if union else 0.0
        return scores


class SaturationSignal:
    """Recall-style: share of a label's distinct evidence tokens matched by
    the state; complements signal 1's presence-count precision."""

    name = "saturation"

    def score(
        self, text: str, tokens: Counter[str], question: ChoiceQuestion
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for label, criteria in question.labels.items():
            evidence = set(tokenize(f"{label} {criteria}"))
            if not evidence:
                scores[label] = 0.0
                continue
            matched = sum(1 for tok in evidence if tokens[tok] > 0)
            scores[label] = matched / len(evidence)
        return scores


class NegationSignal:
    """Penalizes labels whose evidence appears inside a negation window
    ("not a billing issue"). Only this signal sees negations — the other
    signals score the raw, un-negated text; the blend is trusted to keep
    prior-pass evidence while this pass subtracts contradicted support.
    Returns all-zero scores when no negation is present (neutral)."""

    name = "negation"

    def score(
        self, text: str, tokens: Counter[str], question: ChoiceQuestion
    ) -> dict[str, float]:
        words = _WORD_RE.findall(text.lower())
        negated: set[str] = set()
        for i, word in enumerate(words):
            if word in _NEGATIONS or word.endswith("n't"):
                window = tokenize(" ".join(words[i + 1:i + 1 + _NEGATION_WINDOW]))
                negated.update(window)
        if not negated:
            return {label: 0.0 for label in question.labels}
        scores: dict[str, float] = {}
        for label, criteria in question.labels.items():
            evidence = set(tokenize(f"{label} {criteria}"))
            present = {tok for tok in evidence if tokens[tok] > 0}
            penalty = len(present & negated)
            scores[label] = EVIDENCE_WEIGHT * (len(present) - penalty)
        return scores
