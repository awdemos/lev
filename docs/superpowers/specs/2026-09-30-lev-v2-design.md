# LEV v2 Design — from pastiche to genuinely useful

Date: 2026-09-30
Status: approved by user ("do all that")

## Problem with v1 (see README "Honest limitations")

The loop recomputes one signal (token overlap), so iteration = self-trust.
Probabilities are uncalibrated, so the threshold is meaningless. The score
path's loop is arithmetic. Engine is toy-grade.

## v2 changes

### 1. Multi-signal late fusion (the loop becomes load-bearing)

Each pass introduces a *different* deterministic scorer; distributions are
fused with the prior pass via the existing log-space blend:

- **signal 1 (pass 1):** token overlap with stemming (current engine).
- **signal 2:** character 3-gram Jaccard between state and each label's
  `label + criteria` text — catches morphology/typos the stemmer mangles.
- **signal 3:** criteria-as-query saturation: for each label, score = share
  of the label's distinct evidence tokens matched by state (recall-style,
  complements signal 1's presence-count precision).
- **signal 4:** negation-aware re-check: detect negated evidence phrases
  ("not a billing issue", "no refund needed") via a small pattern list;
  penalize labels whose evidence appears negated in the state.

Scorer protocol: `score(state_tokens, question) -> dict[label, float]`.
Passes run until confidence ≥ threshold, budget exhausted, or signals
exhausted (whichever first). `iterations` now means "how many signals were
needed" — a real cost/ambiguity signal. When signals run out but budget
remains, remaining passes reuse the last signal (prior-blend sharpening),
preserving v1 behavior as the floor.

### 2. Calibration (honest probabilities)

Platt scaling on the top-1 margin. `lev calibrate data.jsonl --out calib.json`
fits `(a, b)` for `p(correct) = sigmoid(a * (top1_logit - top2_logit) + b)`
via simple gradient descent on a labeled dataset:

```jsonl
{"text": "...", "question": {"type": "choice", ...}, "expected": "billing"}
```

`classify(..., calibration=calib.json)` reports calibrated confidence.
Without a calibration file, raw (uncalibrated) confidence is reported and
labeled as such in the result (`calibrated: bool` field). Score questions
are calibrated by mapping |score - expected_score| style labels — v2 keeps
calibration to choice questions only (score confidence remains lexical).

### 3. ONNX embedding backend (optional extra)

`pip install lev[onnx]` enables a sentence-embedding scorer: state and each
label's criteria are embedded with a small local model
(onnx-community/all-MiniLM-L6-v2, ~22M params, downloaded once) and scored
by cosine similarity, replacing signals 1–3 with one high-quality signal
(negation pass still applies). Backend selection: `classify(..., backend="heuristic"|"onnx"|"auto")`; `auto` uses ONNX if installed and the model is available, else heuristic. Heuristic remains the default with zero dependencies.

### 4. Routing API (the genuinely useful niche)

```python
from lev import decide

d = decide(text, question, threshold=0.9)   # uses calibration if provided
d.action        # "auto" | "escalate"
d.label         # set when action == "auto"
d.shortlist     # top-2 (label, calibrated prob) when action == "escalate"
d.iterations, d.confidence, d.calibrated
```

`action == "auto"` only when calibrated confidence ≥ threshold; escalate
otherwise. This is the tier-1 triage contract.

## Non-goals

No training of embeddings, no cloud calls, no async, no multi-label
questions, no score-question calibration.

## Compatibility

Public API from v1 (`classify`, `question_from_dict`, types, CLI) unchanged
in signature except new optional kwargs. Version bump to 0.2.0. README
"Honest limitations" updated to reflect what changed and what remains true.
