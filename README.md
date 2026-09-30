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
softmax, so the loop genuinely refines the distribution and the iteration
count reflects input ambiguity.
