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


def test_cli_backend_onnx_missing_exits_nonzero(tmp_path, monkeypatch, capsys):
    import lev.onnx_backend

    def _boom():
        raise RuntimeError("install lev[onnx] to use the onnx backend")

    monkeypatch.setattr(lev.onnx_backend, "_load_model", _boom)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({
        "questions": [{"type": "choice", "question": "q?",
                       "labels": {"a": "words", "b": "other"}}]
    }))
    import lev.cli

    rc = lev.cli.main(["classify", "hello", "--questions", str(questions),
                       "--backend", "onnx"])
    assert rc == 1
    assert "lev: error:" in capsys.readouterr().err
