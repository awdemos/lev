"""Tier-1 triage: classify, then route to auto-handling or escalation.

Uncalibrated confidence must never auto-route — only a fitted calibration
makes the threshold meaningful.
"""
from dataclasses import dataclass

from lev.calibration import Calibration
from lev.loop import classify
from lev.signals import Signal
from lev.types import ChoiceQuestion, ScoreQuestion


@dataclass(frozen=True)
class Decision:
    action: str  # "auto" | "escalate"
    label: str | None
    shortlist: tuple[tuple[str, float], ...]  # top-2 when escalating, else ()
    confidence: float
    calibrated: bool
    margin: float
    iterations: int


def decide(
    text: str,
    question: ChoiceQuestion | ScoreQuestion,
    *,
    threshold: float = 0.9,
    max_iters: int = 5,
    calibration: Calibration | None = None,
    backend: str | None = None,
    signals: tuple[Signal, ...] | None = None,
) -> Decision:
    """Classify and route. action == 'auto' only when calibrated confidence
    meets the threshold; anything uncalibrated or below it escalates with a
    top-2 shortlist."""
    result = classify(
        text,
        question,
        threshold=threshold,
        max_iters=max_iters,
        signals=signals,
        calibration=calibration,
        backend=backend,
    )
    if result.calibrated and result.confidence >= threshold:
        return Decision(
            action="auto",
            label=result.label,
            shortlist=(),
            confidence=result.confidence,
            calibrated=True,
            margin=result.margin,
            iterations=result.iterations,
        )
    shortlist = tuple(
        sorted(result.distribution.items(), key=lambda kv: kv[1], reverse=True)[:2]
    )
    return Decision(
        action="escalate",
        label=None,
        shortlist=shortlist,
        confidence=result.confidence,
        calibrated=result.calibrated,
        margin=result.margin,
        iterations=result.iterations,
    )
