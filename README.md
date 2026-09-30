# LEV — Loop-Evaluated Verdict

*Jev decides once (if). LEV decides until it's sure (while).*

LEV is a Jev-inspired zero-shot classifier. Give it a state (text) and typed
questions (choice or score); instead of answering in a single forward pass,
LEV re-scores in a `while confidence < threshold` loop — each pass fusing a
*different* deterministic signal (token overlap, char 3-gram Jaccard,
evidence saturation, negation penalty; or a local ONNX embedding model) with
the prior distribution — and reports how many signals it needed. Probabilities
can be made honest with Platt scaling (`lev calibrate`), and a routing API
(`decide`) turns calibrated confidence into auto/escalate triage. Obvious
input converges in one pass; ambiguous input burns iteration budget.

## Usage

```python
from lev import classify, ChoiceQuestion

q = ChoiceQuestion(
    question="Which department should handle this?",
    labels={
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else",
    },
)
result = classify(
    "Hi, we were billed twice for March. Please refund the duplicate.",
    q,
)
print(result.label, result.confidence, result.iterations)
```

CLI:

```bash
uv run lev classify "we were billed twice, please refund" \
    --questions questions.json --threshold 0.9
```

where `questions.json` is:

```json
{
  "questions": [
    {
      "type": "choice",
      "question": "Which department should handle this?",
      "labels": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else"
      }
    }
  ]
}
```

## Design

Deterministic heuristic engine by default — no model, no network. Choice
evidence is token overlap between the state and each label's criteria; each
loop pass blends the prior pass's log-probabilities back into the raw scores
before a softmax. The iteration count reflects how ambiguous the input is to
the heuristics, not how much LEV "thought."

Since v2 the loop is load-bearing: pass 1 is token overlap (the v1 signal,
unchanged), pass 2 is character 3-gram Jaccard (catches typos and morphology
the stemmer mangles — "refnd my invoiice" routes correctly), pass 3 is
criteria-as-query saturation (recall-style), and pass 4 penalizes negated
evidence ("not a billing issue"). When the signals run out, the last signal
keeps blending until it reaches a fixed point, so v1's sharpening behavior
remains the floor.

### Calibration

`lev calibrate data.jsonl --out calib.json` fits Platt scaling — `p(correct)
= sigmoid(a·margin + b)` — on labeled records via plain gradient descent,
using the top-1 minus top-2 margin of the final blend. Without a calibration
file the raw (uncalibrated) confidence is reported and `ChoiceResult.calibrated`
is `False`.

### Routing

```python
from lev import decide

d = decide(text, question, threshold=0.9, calibration=calib)
d.action      # "auto" | "escalate"
d.label       # set when action == "auto"
d.shortlist   # top-2 (label, prob) when escalating, else ()
```

`action == "auto"` only when *calibrated* confidence meets the threshold —
uncalibrated confidence never auto-routes; it always escalates. Below
threshold (but calibrated) it escalates with the top-2 shortlist.

### ONNX backend (optional)

`uv pip install 'lev[onnx]'` enables a sentence-embedding signal
(`onnx-community/all-MiniLM-L6-v2-ONNX`, ~22M params, downloaded once) that
replaces signals 1–3 with cosine similarity over mean-pooled embeddings;
the negation pass still applies. Use `classify(..., backend="onnx")`, or
`backend="auto"` to probe and fall back to the heuristic chain.

Score questions measure **urgency/intensity**, not sentiment: an urgent,
angry, or broken-outage message scores high; a calm, low-stakes message
scores low. The confidence reported is the true lexical agreement — LEV
never inflates it to make the loop look productive.

## Honest limitations

LEV is a pastiche of the Jev-style classifier hype, not a production tool.
In the spirit of the project, here is what it actually does:

- **The heuristic signals are shallow.** Token overlap, char n-grams, and
  saturation are lexical tricks. They are deterministic, fast, and much
  better than v1's single signal — and they still do not understand the
  text. The ONNX backend understands it slightly more, but only as much as
  a 22M-param MiniLM can.
- **The negation list is small.** A dozen-ish patterns and a 5-token window;
  "it's not that we don't want no refund" will not end well.
- **Calibration is only as good as your labeled data.** `lev calibrate`
  fits two numbers by gradient descent; a handful of records gives a
  decorative sigmoid, not a reliable probability. Routing decisions on thin
  calibration data are yours to own.
- **The score path's loop is arithmetic in a while-loop costume.** After an
  audit fix, it is literally `iterations = 1 if conf >= threshold else
  max_iters` — kept for contract honesty, but it performs no iteration.
- **The stemmer is a toy.** It handles common cases and a hand-written
  override list, and it mangles some words ("movies" → "movy" is fixed;
  others aren't).
- **The useful output is `iterations` (now: signals-used), not the verdict.**
  The one genuinely interesting idea here is that ambiguity becomes
  observable: a real Jev can't tell you an input was hard, LEV can. Whether
  that is worth a whole classifier is a question the README refuses to answer.

If you want a fast, cheap, calibrated zero-shot classifier, use Jev or
[laya](https://github.com/NandhaKishorM/laya). If you want to see the
"deliberation" trend taken to its logical conclusion, use LEV.
