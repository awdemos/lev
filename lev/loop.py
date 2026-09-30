from collections import Counter

from lev.engine import blend, confidence, raw_choice_scores, tokenize
from lev.types import (
    ChoiceQuestion,
    ChoiceResult,
    ScoreQuestion,
    ScoreResult,
)

_POSITIVE = tokenize(
    "urgent immediately critical asap emergency severe serious important "
    "deadline broken outage down fail failure angry furious terrible awful "
    "love excellent great amazing wonderful happy pleased perfect brilliant"
)
_NEGATIVE = tokenize(
    "later maybe eventually casual minor cosmetic nice fine okay tolerable "
    "someday whenever slow mild slight meh bad poor useless broken"
)


def classify(
    state: str,
    question: ChoiceQuestion | ScoreQuestion,
    threshold: float = 0.9,
    max_iters: int = 5,
) -> ChoiceResult | ScoreResult:
    if isinstance(question, ChoiceQuestion):
        return _classify_choice(state, question, threshold, max_iters)
    return _classify_score(state, question, threshold, max_iters)


def _classify_choice(
    state: str,
    question: ChoiceQuestion,
    threshold: float,
    max_iters: int,
) -> ChoiceResult:
    tokens = tokenize(state)
    raw = raw_choice_scores(tokens, question.labels)
    prior: dict[str, float] | None = None
    iterations = 0
    dist = blend(prior, raw)
    while confidence(dist) < threshold and iterations < max_iters:
        prior = dist
        dist = blend(prior, raw)
        iterations += 1
    label = max(dist, key=dist.get)
    return ChoiceResult(
        label=label,
        distribution=dist,
        confidence=confidence(dist),
        iterations=max(iterations, 1),
        question=question.question,
    )


def _classify_score(
    state: str,
    question: ScoreQuestion,
    threshold: float,
    max_iters: int,
) -> ScoreResult:
    tokens = tokenize(state)
    pos = sum(min(n, tokens[t]) for t, n in _POSITIVE.items())
    neg = sum(min(n, tokens[t]) for t, n in _NEGATIVE.items())
    total = pos + neg
    if total == 0:
        score, conf = 50, 0.1
    else:
        score = round(50 + 50 * (pos - neg) / total)
        conf = abs(pos - neg) / total
    iterations = 1
    # The lexical score is a fixed point; loop only to honor the contract.
    while conf < threshold and iterations < max_iters and total > 0:
        conf = min(1.0, conf + 0.25)
        iterations += 1
    return ScoreResult(
        score=score,
        confidence=conf,
        iterations=iterations,
        question=question.question,
    )
