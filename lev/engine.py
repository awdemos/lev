import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset({"a", "an", "the", "of", "to", "in", "for", "and", "or", "is", "it", "we", "were"})


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    return tok


def tokenize(text: str) -> Counter:
    return Counter(
        _stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS
    )


def softmax(scores: dict[str, float]) -> dict[str, float]:
    peak = max(scores.values())
    exps = {k: math.exp(v - peak) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def confidence(dist: dict[str, float]) -> float:
    return max(dist.values())


def raw_choice_scores(state: Counter, labels: dict[str, str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, criteria in labels.items():
        evidence = tokenize(f"{label} {criteria}")
        scores[label] = 3.0 * float(
            sum(min(count, state[tok]) for tok, count in evidence.items())
        )
    return scores


def blend(
    prior: dict[str, float] | None,
    raw: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    if prior is None:
        return softmax(raw)
    combined = {
        k: raw[k] + alpha * math.log(max(prior[k], 1e-9)) for k in raw
    }
    return softmax(combined)
