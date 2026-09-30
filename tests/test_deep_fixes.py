import json
import math
import subprocess
import sys

import pytest

from lev.engine import raw_choice_scores, tokenize
from lev.loop import classify
from lev.types import ScoreQuestion

URGENT_Q = ScoreQuestion(question="how urgent?")


def test_single_urgent_word_has_volume_scaled_confidence():
    result = classify("urgent", URGENT_Q)
    assert result.score == 100
    assert result.confidence == pytest.approx(1 / 3)


def test_repeated_urgent_word_reaches_full_confidence():
    result = classify("urgent " * 5, URGENT_Q)
    assert result.confidence == 1.0


def test_stemmer_plural_singular_parity():
    assert tokenize("crashes") == {"crash": 1}
    assert tokenize("statuses") == {"status": 1}
    assert tokenize("movies") == {"movie": 1}
    assert tokenize("analyses") == {"analysis": 1}
    assert tokenize("billing") == {"bill": 1}
    assert tokenize("billed") == {"bill": 1}


def test_cli_rejects_non_dict_top_level(tmp_path):
    for payload in ([], "hello", 42):
        questions = tmp_path / "questions.json"
        questions.write_text(json.dumps(payload))
        proc = subprocess.run(
            [sys.executable, "-m", "lev.cli", "classify", "hi",
             "--questions", str(questions)],
            capture_output=True, text=True, cwd=".",
        )
        assert proc.returncode == 1, payload
        assert proc.stderr.startswith("lev: error:")
        assert "Traceback" not in proc.stderr


POS_20 = ("urgent immediately critical asap emergency severe serious important "
          "deadline broken outage down fail failure angry furious terrible awful "
          "love excellent")
NEG_12 = ("later maybe eventually casual minor cosmetic nice fine okay "
          "tolerable someday whenever")


def test_score_rounds_half_up_not_bankers():
    result = classify(f"{POS_20} {NEG_12}", URGENT_Q)
    # 50 + 50 * 8/32 = 62.5 -> must round to 63
    assert result.score == 63


def test_criteria_duplication_does_not_inflate_choice_scores():
    labels = {"A": "refund refund", "B": "billing"}
    single = raw_choice_scores(tokenize("refund"), labels)
    repeated = raw_choice_scores(tokenize("refund refund refund"), labels)
    assert repeated == single
    assert single["A"] == 3.0
