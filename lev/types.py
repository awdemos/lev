from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChoiceQuestion:
    question: str
    labels: dict[str, str]


@dataclass(frozen=True)
class ScoreQuestion:
    question: str
    low_anchor: str = "worst case"
    high_anchor: str = "best case"


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
