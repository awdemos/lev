from lev.types import ChoiceQuestion, ScoreQuestion


def question_from_dict(data: dict) -> ChoiceQuestion | ScoreQuestion:
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
            low_anchor=str(data.get("low_anchor", "worst case")),
            high_anchor=str(data.get("high_anchor", "best case")),
        )
    raise ValueError(f"unknown type: {qtype!r}")
