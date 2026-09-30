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

_ONNX_CHAIN: tuple[Signal, ...] | None = None
_ONNX_PROBE: bool | None = None


def _onnx_chain() -> tuple[Signal, ...]:
    """ONNX backend: one embedding signal replaces signals 1-3; the
    negation pass still applies."""
    global _ONNX_CHAIN
    if _ONNX_CHAIN is None:
        from lev.onnx_backend import OnnxEmbeddingSignal
        from lev.signals import NegationSignal

        _ONNX_CHAIN = (OnnxEmbeddingSignal(), NegationSignal())
    return _ONNX_CHAIN


def _onnx_available() -> bool:
    """True if the extras are installed and the model can be loaded."""
    global _ONNX_PROBE
    if _ONNX_PROBE is None:
        try:
            _onnx_chain()[0].score("", {}, ChoiceQuestion(question="", labels={"a": "b"}))
            _ONNX_PROBE = True
        except Exception:
            _ONNX_PROBE = False
    return _ONNX_PROBE


def classify(
    text: str,
    question: ChoiceQuestion | ScoreQuestion,
    threshold: float = 0.9,
    max_iters: int = 5,
    signals: tuple[Signal, ...] | None = None,
    calibration: Calibration | None = None,
    backend: str | None = None,
) -> ChoiceResult | ScoreResult:
    """Classify text against a typed question via a confidence-gated loop.

    Raises ValueError if threshold is not in [0, 1], max_iters < 1, a
    choice question has empty labels, backend is not heuristic/onnx/auto,
    or both signals and backend are passed.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if signals is not None and backend is not None:
        raise ValueError("signals and backend are mutually exclusive")
    if signals is None:
        backend = backend or "heuristic"
        if backend == "heuristic":
            signals = DEFAULT_SIGNALS
        elif backend == "onnx":
            signals = _onnx_chain()
        elif backend == "auto":
            signals = _onnx_chain() if _onnx_available() else DEFAULT_SIGNALS
        else:
            raise ValueError(f"unknown backend: {backend!r}")
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
