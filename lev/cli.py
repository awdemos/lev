import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from lev.calibration import Calibration
from lev.loop import classify
from lev.router import question_from_dict
from lev.types import ChoiceQuestion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lev",
        description="Loop-Evaluated Verdict: a Jev-style classifier that deliberates in while-loops.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    classify_parser = sub.add_parser("classify", help="classify a state string")
    classify_parser.add_argument("state", help="the text to classify")
    classify_parser.add_argument("--questions", required=True, help="path to questions JSON")
    classify_parser.add_argument("--threshold", type=float, default=0.9)
    classify_parser.add_argument("--max-iters", type=int, default=5)
    classify_parser.add_argument("--calibration", help="path to a calibration JSON file")
    calibrate_parser = sub.add_parser("calibrate", help="fit Platt scaling from labeled JSONL")
    calibrate_parser.add_argument("data", help="path to JSONL records: text, question, expected")
    calibrate_parser.add_argument("--out", required=True, help="path to write calibration JSON")
    args = parser.parse_args(argv)

    try:
        if args.command == "calibrate":
            return _calibrate(args.data, args.out)
        payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("questions file must contain a JSON object")
        questions = payload.get("questions")
        if not isinstance(questions, list) or not all(
            isinstance(q, dict) for q in questions
        ):
            raise ValueError("questions file must contain a 'questions' list")
        calibration = (
            Calibration.load(args.calibration) if args.calibration else None
        )
        results = [
            classify(
                args.state,
                question_from_dict(q),
                args.threshold,
                args.max_iters,
                calibration=calibration,
            )
            for q in questions
        ]
    except (ValueError, OSError) as exc:
        print(f"lev: error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(json.dumps(asdict(result), indent=2))
    return 0


def _calibrate(data_path: str, out_path: str) -> int:
    """Fit Platt scaling on (margin, was_correct) from labeled records."""
    records: list[tuple[float, bool]] = []
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = question_from_dict(row["question"])
        if not isinstance(question, ChoiceQuestion):
            continue  # v2 calibrates choice questions only
        result = classify(
            str(row["text"]), question, threshold=1.0, max_iters=5
        )
        records.append((result.margin, result.label == str(row["expected"])))
    if not records:
        raise ValueError("calibration data contains no choice records")
    Calibration.fit(records).save(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
