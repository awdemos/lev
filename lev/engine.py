import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
STOPWORDS = frozenset({"a", "an", "the", "of", "to", "in", "for", "and", "or", "is", "it", "we", "were"})


_OVERRIDES = {
    "series": "series",
    "movies": "movie",
    "analyses": "analysis",
    "status": "status",
    "statuses": "status",
}


# (suffix, min token length, cut, reject if stem ends with)
_RULES = (
    ("ies", 4, 3, "y", None),
    ("es", 4, 2, "", "s"),
    ("ing", 5, 3, "", None),
    ("s", 3, 1, "", None),
    ("ed", 4, 2, "", None),
)

_EXEMPT_ENDINGS = ("ss", "us", "is")
_MIN_STEM_LEN = 3


def _stem(tok: str) -> str:
    if tok in _OVERRIDES:
        return _OVERRIDES[tok]
    if tok.endswith(_EXEMPT_ENDINGS):
        return tok
    for suffix, min_len, cut, add, reject in _RULES:
        if len(tok) > min_len and tok.endswith(suffix):
            stem = tok[:-cut] + add
            if reject and stem.endswith(reject):
                return tok
            return stem if len(stem) >= _MIN_STEM_LEN else tok
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
            sum(1 for tok in evidence if state[tok] > 0)
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
