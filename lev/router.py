from lev.types import (
    DEFAULT_HIGH_ANCHOR,
    DEFAULT_LOW_ANCHOR,
    ChoiceQuestion,
    ScoreQuestion,
)


def question_from_dict(data: dict[str, object]) -> ChoiceQuestion | ScoreQuestion:
    """Build a typed question from a dict.

    Raises ValueError for an unknown 'type', missing/empty 'labels' on a
    choice question, or fewer than two labels.
    """
    qtype = data.get("type")
    if qtype == "choice":
        labels = data.get("labels")
        if not isinstance(labels, dict) or not labels:
            raise ValueError("choice questions require a non-empty 'labels' map")
        if len(labels) < 2:
            raise ValueError("choice questions require at least two labels")
        return ChoiceQuestion(question=str(data.get("question", "")), labels=labels)
    if qtype == "score":
        return ScoreQuestion(
            question=str(data.get("question", "")),
            low_anchor=str(data.get("low_anchor", DEFAULT_LOW_ANCHOR)),
            high_anchor=str(data.get("high_anchor", DEFAULT_HIGH_ANCHOR)),
        )
    raise ValueError(f"unknown type: {qtype!r}")
