import argparse
import json
import sys
from pathlib import Path

from lev.loop import classify
from lev.router import question_from_dict


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
    args = parser.parse_args(argv)

    try:
        payload = json.loads(Path(args.questions).read_text())
        results = [
            classify(args.state, question_from_dict(q), args.threshold, args.max_iters)
            for q in payload["questions"]
        ]
    except (ValueError, KeyError, OSError) as exc:
        print(f"lev: error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
