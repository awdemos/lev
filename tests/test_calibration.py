import json
import math
import subprocess
import sys

import pytest

from lev.calibration import Calibration
from lev.loop import classify
from lev.types import ChoiceResult, ChoiceQuestion

QUESTION = ChoiceQuestion(
    question="Which department should handle this?",
    labels={
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else",
    },
)


def test_sigmoid_math():
    calib = Calibration(a=2.0, b=-1.0)
    assert calib.p(0.5) == pytest.approx(1 / (1 + math.exp(-(2 * 0.5 - 1))))
    assert calib.p(0.0) == pytest.approx(1 / (1 + math.e))


def test_fit_recovers_extremes():
    records = [(5.0, True), (4.0, True), (3.0, True),
               (-3.0, False), (-4.0, False), (-5.0, False)]
    calib = Calibration.fit(records)
    assert calib.p(5.0) > 0.9
    assert calib.p(-5.0) < 0.1
    assert calib.p(5.0) > calib.p(0.0) > calib.p(-5.0)


def test_save_load_roundtrip(tmp_path):
    calib = Calibration(a=1.5, b=-0.25)
    path = tmp_path / "calib.json"
    calib.save(path)
    assert Calibration.load(path) == calib
    assert json.loads(path.read_text()) == {"a": 1.5, "b": -0.25}


def test_result_defaults_keep_compat():
    r = ChoiceResult(label="a", distribution={"a": 1.0}, confidence=1.0, iterations=1)
    assert r.calibrated is False
    assert r.margin == 0.0


def test_classify_with_calibration_reports_calibrated_confidence():
    calib = Calibration(a=1.0, b=0.0)
    result = classify("please refund my invoice", QUESTION, calibration=calib)
    assert result.calibrated is True
    assert result.confidence == pytest.approx(calib.p(result.margin))
    raw = classify("please refund my invoice", QUESTION)
    assert raw.calibrated is False
    assert result.margin > 0


def test_calibrated_ordering_matches_empirical_accuracy():
    # Fit on synthetic records where large margins are correct.
    records = [(8.0, True), (6.0, True), (1.0, False), (-2.0, False)]
    calib = Calibration.fit(records)
    easy = classify("please refund my duplicate invoice payment", QUESTION,
                    calibration=calib)
    hard = classify("invoice bug", QUESTION, calibration=calib)
    assert easy.confidence > hard.confidence


def test_cli_calibrate_and_classify(tmp_path):
    data = tmp_path / "data.jsonl"
    q = {"type": "choice", "question": "Which department?",
         "labels": {"billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, system errors"}}
    rows = [
        {"text": "please refund my invoice", "question": q, "expected": "billing"},
        {"text": "the system error broke everything", "question": q, "expected": "technical"},
        {"text": "invoice is broken, system error", "question": q, "expected": "billing"},
        {"text": "refund the payment now", "question": q, "expected": "billing"},
        {"text": "outage caused the failure", "question": q, "expected": "technical"},
        {"text": "error on the invoice page", "question": q, "expected": "technical"},
    ]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "calib.json"
    proc = subprocess.run(
        [sys.executable, "-m", "lev.cli", "calibrate", str(data),
         "--out", str(out)], capture_output=True, text=True, cwd=".")
    assert proc.returncode == 0, proc.stderr
    calib = Calibration.load(out)
    assert isinstance(calib.a, float) and isinstance(calib.b, float)

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [q]}))
    proc = subprocess.run(
        [sys.executable, "-m", "lev.cli", "classify", "please refund my invoice",
         "--questions", str(questions), "--calibration", str(out)],
        capture_output=True, text=True, cwd=".")
    assert proc.returncode == 0, proc.stderr
    assert '"calibrated": true' in proc.stdout
