import json
import subprocess
import sys


def _run(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "lev.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_classify_choice(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "type": "choice",
                        "question": "Which department?",
                        "labels": {
                            "billing": "invoices, payments, refunds",
                            "other": "everything else",
                        },
                    }
                ]
            }
        )
    )
    proc = _run(
        ["classify", "please refund my invoice", "--questions", str(questions)],
        cwd=".",
    )
    assert proc.returncode == 0, proc.stderr
    assert '"label": "billing"' in proc.stdout
    assert '"iterations"' in proc.stdout


def test_bad_questions_file_exits_nonzero(tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [{"type": "rank"}]}))
    proc = _run(["classify", "hello", "--questions", str(questions)], cwd=".")
    assert proc.returncode == 1
