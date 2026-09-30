from dataclasses import dataclass

DEFAULT_LOW_ANCHOR = "worst case"
DEFAULT_HIGH_ANCHOR = "best case"


@dataclass(frozen=True)
class ChoiceQuestion:
    question: str
    labels: dict[str, str]


@dataclass(frozen=True)
class ScoreQuestion:
    question: str
    low_anchor: str = DEFAULT_LOW_ANCHOR
    high_anchor: str = DEFAULT_HIGH_ANCHOR


@dataclass(frozen=True)
class ChoiceResult:
    label: str
    distribution: dict[str, float]
    confidence: float
    iterations: int
    question: str = ""


@dataclass(frozen=True)
class ScoreResult:
    score: int
    confidence: float
    iterations: int
    question: str = ""
