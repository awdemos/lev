# LEV Design — a Jev-inspired classifier that deliberates in while-loops

Date: 2026-09-30
Status: approved by user

## Concept

Jev (TypeSafe AI) is a zero-shot classifier: give it a state and a typed
question (choice or score), and it answers in a single forward pass with a
calibrated probability. Its open-source counterpart is
[laya](https://github.com/NandhaKishorM/laya).

LEV ("Loop-Evaluated Verdict") keeps Jev's typed-question interface but
replaces the one-shot `if` decision with a **while-loop of refinement**:
LEV re-scores the state, feeding back the previous pass's distribution as
evidence, and only commits when its confidence crosses a threshold or an
iteration budget is exhausted. *Jev decides once (if); LEV decides until it
is sure (while).*

## Loop mechanics (chosen approach)

Confidence-gated refinement. Pseudocode:

```
dist = score(state, question, prior=None)
while confidence(dist) < threshold and i < max_iters:
    dist = score(state, question, prior=dist)
    i += 1
```

Each pass is deterministic and pure; the loop is the load-bearing feature.
The number of iterations spent is returned and observable — ambiguous input
spends more iterations, obvious input converges in one pass.

Rejected alternatives:
- **Annealing loop** — sharpens probabilities without new evidence; fake
  confidence, contradicts the calibrated spirit.
- **Self-critique loop** — needs an LLM to be meaningful.

## Backend

Deterministic heuristic engine. No ML model, no network. Label evidence is
scored by token-overlap between the input state and each label's criteria
text, plus a feedback term from the prior pass's distribution. Pure
functions throughout; fully unit-testable offline.

Rejected alternatives: ONNX local model (heavier deps), pluggable backends
(YAGNI for v1).

## Question types

- `ChoiceQuestion`: `{question, labels: {name: criteria}}` → answer label +
  full probability distribution + confidence + iterations.
- `ScoreQuestion`: `{question, anchors?}` → score 0–100 + confidence +
  iterations. Sentiment-ish lexical scoring mapped onto the 0–100 range.

## Package layout

New directory `/var/home/a/code/lev/`, uv-managed, Python ≥3.10, stdlib-only.

- `lev/types.py` — `ChoiceQuestion`, `ScoreQuestion`, result dataclasses.
- `lev/engine.py` — pure scoring functions (tokenization, overlap scoring,
  prior-feedback blending).
- `lev/loop.py` — `classify(state, question, threshold=0.9, max_iters=5)`;
  the while-loop lives here.
- `lev/router.py` — validates question dicts and dispatches by `type`.
- `lev/cli.py` — `lev classify "text" --questions questions.json
  [--threshold 0.9] [--max-iters 5]`.
- `tests/` — engine unit tests + loop-behavior tests (converges fast on
  obvious input, respects budget on ambiguous input, threshold/edge cases).
- `README.md` — pitch and usage.

## Explicit non-goals (v1)

No HTTP server, no ONNX/transformers, no multilingual checkpoints, no
fine-tuning. Library + CLI only.
