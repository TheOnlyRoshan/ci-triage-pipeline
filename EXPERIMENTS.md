# Experiments

Running log of measured decisions. Every entry states the question,
the setup, the numbers, and what was decided.

## Conventions

- All experiments run on the 27-example dev-eval split (seed 42).
- The 10 final-check IDs are held out and untouched until Phase 1c is complete.
- Metrics reported per-class (precision/recall), never as a single accuracy number.
- Baseline for comparison: random (25%, 4 balanced classes) and majority-class.
- Dataset is small (n=27) — all numbers carry wide confidence intervals.

---

## E1: Preprocessing — strip env block / pip output?

**Question:** Does removing the runner env block and pip install output
from the step log improve or degrade classification?

**Hypothesis:** No gain on flaky/transient/regression; possible loss on
infra, where env vars (pythonLocation, LD_LIBRARY_PATH) are diagnostic.

**Setup:**

- Model: <TBD>
- Prompt version: <TBD>
- Variants: strip_env_block ∈ {false, true} × strip_pip_output ∈ {false, true}
- Fixed: temperature 0, structural strips always on

**Results:**

| Variant             | flaky P/R | infra P/R | regression P/R | transient P/R | Overall |
|---------------------|-----------|-----------|----------------|---------------|---------|
| both off (baseline) |           |           |                |               |         |
| env on              |           |           |                |               |         |
| pip on              |           |           |                |               |         |
| both on             |           |           |                |               |         |

**Decision:** <pending>

**Notes:** Whatever is chosen must be identical in eval and production —
a flag that differs between the two is train/serve skew.

---

## E2: Three-model comparison

**Question:** Which model gives the best cost/accuracy trade-off as labeler?

**Setup:**

- Models: <TBD>
- Prompt version: fixed across all three
- Preprocessing: whatever E1 decided

**Results:**

| Model | flaky P/R | infra P/R | regression P/R | transient P/R | Overall | Cost/1k | Latency |
|-------|-----------|-----------|----------------|---------------|---------|---------|---------|

**Decision:** <pending>

---

## Template for new entries

## E<n>: <title>

**Question:**
**Hypothesis:**
**Setup:** model, prompt version, what varies, what's fixed
**Results:** table, per-class
**Decision:**
**Notes:**