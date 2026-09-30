# LEV — Loop-Evaluated Verdict

*Jev decides once (if). LEV decides until it's sure (while).*

LEV is a Jev-inspired zero-shot classifier. Give it a state (text) and typed
questions (choice or score); instead of answering in a single forward pass,
LEV re-scores in a `while confidence < threshold` loop, feeding each pass's
probability distribution back in as evidence, and reports how many
iterations it spent deciding. Obvious input converges in one pass; ambiguous
input burns iteration budget.

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

Deterministic heuristic engine — no model, no network. Choice evidence is
token overlap between the state and each label's criteria; each loop pass
blends the prior pass's log-probabilities back into the raw scores before a
softmax. The iteration count reflects how ambiguous the input is to the
heuristic, not how much LEV "thought."

Score questions measure **urgency/intensity**, not sentiment: an urgent,
angry, or broken-outage message scores high; a calm, low-stakes message
scores low. The confidence reported is the true lexical agreement — LEV
never inflates it to make the loop look productive.

## Honest limitations

LEV is a pastiche of the Jev-style classifier hype, not a production tool.
In the spirit of the project, here is what it actually does:

- **The deliberation is self-trust, not new evidence.** The loop computes a
  fixed point of `softmax(raw + α·log prior)` — it re-reads its own first
  guess with growing confidence. No new information enters per pass, so the
  loop measures how willing LEV is to agree with itself, not how thoroughly
  it has reasoned. Genuine iterative refinement needs new signal each pass;
  LEV has none.
- **The probabilities are not calibrated.** They come from token overlap and
  were never trained, so `confidence ≥ threshold` is a bar over made-up
  numbers. The one thing that made Jev more than "just a classifier" —
  calibrated probabilities — is exactly the part LEV cannot replicate
  without a trained model.
- **The score path's loop is arithmetic in a while-loop costume.** After an
  audit fix, it is literally `iterations = 1 if conf >= threshold else
  max_iters` — kept for contract honesty, but it performs no iteration.
- **The stemmer is a toy.** It handles common cases and a hand-written
  override list, and it mangles some words ("movies" → "movy" is fixed;
  others aren't).
- **The useful output is `iterations`, not the verdict.** The one genuinely
  interesting idea here is that ambiguity becomes observable: a real Jev
  can't tell you an input was hard, LEV can. Whether that is worth a whole
  classifier is a question the README refuses to answer.

If you want a fast, cheap, calibrated zero-shot classifier, use Jev or
[laya](https://github.com/NandhaKishorM/laya). If you want to see the
"deliberation" trend taken to its logical conclusion, use LEV.
