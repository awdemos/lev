import math
import re
from collections import Counter
from functools import lru_cache

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
STOPWORDS = frozenset({"a", "an", "the", "of", "to", "in", "for", "and", "or", "is", "it", "we", "were"})
EVIDENCE_WEIGHT = 3.0

_OVERRIDES = {
    "series": "series",
    "movies": "movie",
    "analyses": "analysis",
    "status": "status",
    "statuses": "status",
    "invoices": "invoice",
}


def _stem(tok: str) -> str:
    if tok in _OVERRIDES:
        return _OVERRIDES[tok]
    if tok.endswith(("ss", "us", "is")):
        return tok
    if len(tok) > 4 and tok.endswith("ies"):
        stem = tok[:-3] + "y"
    elif len(tok) > 4 and tok.endswith("es"):
        stem = tok[:-2]
        if stem.endswith("s"):
            return tok
    elif len(tok) > 5 and tok.endswith("ing"):
        stem = tok[:-3]
    elif len(tok) > 3 and tok.endswith("s"):
        stem = tok[:-1]
    elif len(tok) > 4 and tok.endswith("ed"):
        stem = tok[:-2]
    else:
        return tok
    return stem if len(stem) >= 3 else tok


def tokenize(text: str) -> Counter[str]:
    return Counter(
        _stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS
    )


_POSITIVE_WORDS = (
    "urgent immediately critical asap emergency severe serious important "
    "deadline broken outage down fail failure angry furious terrible awful "
    "love excellent great amazing wonderful happy pleased perfect brilliant"
)
_NEGATIVE_WORDS = (
    "later maybe eventually casual minor cosmetic nice fine okay tolerable "
    "someday whenever slow mild slight meh bad poor useless"
)


@lru_cache(maxsize=1)
def _lexicons() -> tuple[frozenset[str], frozenset[str]]:
    """Stemmed urgency lexicons, built lazily so tokenize()/_stem changes
    always take effect at call time rather than being snapshotted at import."""
    return frozenset(tokenize(_POSITIVE_WORDS)), frozenset(tokenize(_NEGATIVE_WORDS))


def softmax(scores: dict[str, float]) -> dict[str, float]:
    peak = max(scores.values())
    exps = {k: math.exp(v - peak) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def confidence(dist: dict[str, float]) -> float:
    return max(dist.values())


def raw_choice_scores(tokens: Counter[str], labels: dict[str, str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, criteria in labels.items():
        evidence = tokenize(f"{label} {criteria}")
        scores[label] = EVIDENCE_WEIGHT * sum(1 for tok in evidence if tokens[tok] > 0)
    return scores


def urgency_score(tokens: Counter[str]) -> tuple[int, float]:
    """Map token counts to an urgency score (0-100) and confidence."""
    pos_words, neg_words = _lexicons()
    pos = sum(tokens[t] for t in pos_words)
    neg = sum(tokens[t] for t in neg_words)
    total = pos + neg
    if total == 0:
        return 50, 0.1
    score = math.floor(50 + 50 * (pos - neg) / total + 0.5)
    conf = (abs(pos - neg) / total) * min(1.0, total / 3.0)
    return score, conf


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
