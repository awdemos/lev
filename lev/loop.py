"""Loop-driven classification: signals are fused until confidence, budget,
or signal exhaustion settles the distribution."""
import math

from lev.calibration import Calibration
from lev.engine import blend, confidence, tokenize, urgency_score
from lev.signals import (
    CharNgramSignal,
    NegationSignal,
    SaturationSignal,
    Signal,
    TokenOverlapSignal,
)
from lev.types import (
    ChoiceQuestion,
    ChoiceResult,
    ScoreQuestion,
    ScoreResult,
)

DEFAULT_SIGNALS: tuple[Signal, ...] = (
    TokenOverlapSignal(),
    CharNgramSignal(),
    SaturationSignal(),
    NegationSignal(),
)


def classify(
    text: str,
    question: ChoiceQuestion | ScoreQuestion,
    threshold: float = 0.9,
    max_iters: int = 5,
    signals: tuple[Signal, ...] = DEFAULT_SIGNALS,
    calibration: Calibration | None = None,
) -> ChoiceResult | ScoreResult:
    """Classify text against a typed question via a confidence-gated loop.

    Raises ValueError if threshold is not in [0, 1], max_iters < 1, or a
    choice question has empty labels.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if isinstance(question, ChoiceQuestion):
        return _classify_choice(text, question, threshold, max_iters, signals, calibration)
    return _classify_score(text, question, threshold, max_iters)


def _classify_choice(
    text: str,
    question: ChoiceQuestion,
    threshold: float,
    max_iters: int,
    signals: tuple[Signal, ...] = DEFAULT_SIGNALS,
    calibration: Calibration | None = None,
) -> ChoiceResult:
    if not question.labels:
        raise ValueError("labels must be non-empty")
    tokens = tokenize(text)
    prior: dict[str, float] | None = None
    dist: dict[str, float] = {}
    combined: dict[str, float] = {}
    iterations = 0
    while True:
        signal = signals[min(iterations, len(signals) - 1)]
        raw = signal.score(text, tokens, question)
        combined = (
            raw if prior is None
            else {k: raw[k] + 0.5 * math.log(max(prior[k], 1e-9)) for k in raw}
        )
        dist = blend(prior, raw)
        iterations += 1
        if confidence(dist) >= threshold or iterations >= max_iters:
            break
        if iterations >= len(signals):
            # Signals exhausted: stop only at a fixed point of the last
            # signal's raw scores; otherwise keep sharpening (v1 floor).
            probe = blend(dist, raw)
            if abs(confidence(probe) - confidence(dist)) < 1e-9:
                break
        prior = dist
    label = max(dist, key=dist.get)
    ranked = sorted(dist, key=dist.get, reverse=True)
    margin = combined[ranked[0]] - combined[ranked[1]]
    calibrated = calibration is not None
    return ChoiceResult(
        label=label,
        distribution=dist,
        confidence=calibration.p(margin) if calibration else confidence(dist),
        iterations=iterations,
        question=question.question,
        calibrated=calibrated,
        margin=margin,
    )


def _classify_score(
    text: str,
    question: ScoreQuestion,
    threshold: float,
    max_iters: int,
) -> ScoreResult:
    score, conf = urgency_score(tokenize(text))
    # The lexical score is a fixed point: the loop only counts how many
    # iterations an unconfirmable score spends failing to reach the
    # threshold; confidence is never inflated.
    iterations = 1 if conf >= threshold else max_iters
    return ScoreResult(
        score=score,
        confidence=conf,
        iterations=iterations,
        question=question.question,
    )
