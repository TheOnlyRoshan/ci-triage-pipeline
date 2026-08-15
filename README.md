# ci-triage-pipeline

Classifies a failed GitHub Actions job into one of four failure categories by
reading its log with an LLM.

CI failures are not all the same problem. A flaky test wastes an engineer's
afternoon; a real regression should block the merge; an infrastructure failure
needs a different team entirely. Triage is the step where someone reads the log
and decides which it is, and it is the step that gets skipped when a pipeline is
red three times a day.

## Categories

| Category | Meaning |
|:-|:-|
| `flaky_test` | Non deterministic failure originating inside the repository: randomness, timing assumptions, test order dependence |
| `genuine_regression` | A real defect in the code under test, deterministic and reproducible |
| `infra` | The CI environment failed to build or configure, so tests never ran |
| `transient` | The environment built, but an external dependency failed during execution |

The two boundaries that matter are `flaky_test` against `transient` (where does
the non determinism originate, inside the repo or outside it) and `transient`
against `infra` (did the environment build at all). Both are stated explicitly
as disambiguation rules in the prompt.

## Current results

Measured on a held out 27 example split, zero shot, temperature 0.0:

```
Class                Precision    Recall        F1   Support
flaky_test                1.00      0.86      0.92         7
genuine_regression        0.71      1.00      0.83        10
infra                     1.00      0.67      0.80         3
transient                 1.00      0.71      0.83         7
Overall accuracy          0.85                            27

Random baseline           0.25
Majority class baseline   0.37
```

Two caveats belong with that number, and both are documented in
[EXPERIMENTS.md](EXPERIMENTS.md):

1. All four errors were predicted as `genuine_regression`. No other pair of
   categories was ever confused. That is a specific, fixable prompt problem
   rather than diffuse noise.
2. The number is inflated. The fault injected tests in the dataset carry
   docstrings that state their own category in plain English, and pytest prints
   test source on failure, so that text reaches the prompt. Quantifying and
   removing that effect is the next piece of work.

## How it works

```
job_id
  |
  v
job_fetcher.get_failed_step()        GitHub API: which step failed, and when
  |
  v
log_fetcher.fetch_job_log()          GitHub API: the whole job log
log_fetcher.extract_step_window()    slice to the failed step by timestamp
  |
  v
log_preprocessor.preprocess_log()    strip noise, cap to head + tail window
  |
  v
prompt_builder.build_prompt()        substitute into versioned template
  |
  v
llm_labeler.label_log()              Anthropic API, parse, validate
  |
  v
label_store.get_or_create()          JSONL cache, keyed on the whole question
  |
  v
LabelResult(label, confidence, rationale)
```

Each stage is a separate module with a narrow contract, so the two that involve
no network access (`log_preprocessor`, `evaluation.metrics`) are testable
against plain string fixtures.

### Isolating the failed step

The `/logs` endpoint returns every step of a job concatenated. Matching on
`##[group]` markers does not work: a step's `name` in job metadata differs from
its marker text in the log ("Run tests" against "##[group]Run pytest"), and one
step can contain several groups plus ungrouped lines, so there is no one to one
mapping.

Slicing by timestamp does work. Every log line is prefixed with an ISO 8601
timestamp, and job metadata gives each step a `started_at` and `completed_at`.
GitHub reports those boundaries only to whole second precision, so the final
`##[error]` line sometimes falls just outside the window and is clipped.
Padding by a second recovers it but also pulls in the entire post job cleanup
block, which would crowd out the actual failure in the tail window downstream.
Precision was chosen over recall.

### The label store

`data/labels.jsonl` is append only, one JSON object per line, keyed on four
fields: `example_id`, `prompt_version`, `model`, and `preprocessing_variant`.

A stored label is reused only when all four match. Changing the model or the
preprocessing flags means a different question was asked, so a stale answer must
not be returned for it. Without `model` in the key, a model comparison would
show two models agreeing perfectly because the second was never actually called.

Nothing is written when labelling fails. Caching a parse error would make the
failure permanent until the file was edited by hand.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file at the project root:

```
GITHUB_TOKEN=<fine grained PAT with Actions:Read>
ANTHROPIC_API_KEY=<key from console.anthropic.com>
```

Clone the dataset into `data/dataset/`, pinned to the SHA in `config.yaml`:

```bash
git clone https://github.com/TheOnlyRoshan/ci-triage-dataset data/dataset
cd data/dataset && git checkout 97167a8d9d4d7976f694decfbefd9d97ddc9ba43
```

## Usage

Triage a single job:

```bash
python -m src.main <job_id>
python -m src.main <job_id> --no_store      # bypass the cache
```

Run the evaluation over a split:

```bash
python -m tools.run_eval
python -m tools.run_eval --split dev_eval --output reports/run.txt
```

Verify the committed split against the dataset and the seed:

```bash
python tools/make_split.py
```

The first evaluation run makes one API call per example. Later runs are served
from `data/labels.jsonl` and are free, unless the model, prompt version, or
preprocessing variant changed.

## Configuration

Every tunable lives in `config.yaml` and is validated into typed Pydantic models
at load time, so a bad regex, an out of range temperature, or an overlapping
split fails at startup rather than midway through a paid run.

| Section | Controls |
|:-|:-|
| `categories` | The four labels, cross checked against `src/categories.py` |
| `github` | Repository and token environment variable name |
| `preprocessing` | Noise patterns, ablation flags, window sizes |
| `prompt` | Template version and directory |
| `llm` | Model, temperature, max tokens |
| `label_store` | Cache file path |
| `split` | Exemplar and final check IDs, and the seed they were drawn with |
| `dataset` | Dataset repository and pinned SHA |

Secrets are never in config. Config holds the environment variable *name*, and
`auth.get_secret()` resolves it at runtime.

## Dataset and splits

49 examples across the four classes, in a
[separate repository](https://github.com/TheOnlyRoshan/ci-triage-dataset) pinned
by SHA. Real CI logs captured through fault injection, plus synthetic logs where
a failure mode was impractical to reproduce.

| Split | Size | Purpose |
|:-|:-|:-|
| `exemplars` | 12 | Reserved as few shot candidates, never evaluated on |
| `dev_eval` | 27 | Every number reported so far |
| `final_check` | 10 | Untouched holdout |

`dev_eval` is derived as the complement of the other two rather than listed in
config, so it cannot drift out of sync with them.

Ground truth reaches evaluation and never the prompt. `Example` carries only
`example_id`, `log_text`, and `true_label`; the sidecar's `injected_fault` field,
which describes the bug in English, is deliberately not loaded.

## Reproducibility

The pieces that make a run repeatable, in rough order of how easy they are to
get wrong:

* Temperature 0.0, and the model pinned in config.
* Prompt text versioned in `prompts/`, with the version recorded on every stored
  label. Editing the prompt invalidates old labels rather than silently mixing
  them.
* Dataset pinned by commit SHA.
* Split selection seeded, and `tools/make_split.py` regenerates it to verify.
  An earlier bug there is worth recording: iterating a `Counter` built from a
  sorted list changed class iteration order, so the same seed produced a
  different split. Fixed by giving each class its own RNG seeded on
  `f"{seed}:{label}"`, which removes iteration order from the inputs entirely.
* Every label persisted with the full four field key that produced it.

## Project layout

```
config.yaml               all tunables, validated at load
prompts/v1.txt            versioned prompt template
data/dataset/             the 49 examples (gitignored, cloned separately)
data/labels.jsonl         every label ever produced
reports/                  saved evaluation reports
EXPERIMENTS.md            measured decisions, one entry per experiment

src/
  categories.py           single definition of the four labels
  auth.py                 environment variable to secret
  config_loader.py        Pydantic models and validators
  main.py                 CLI entry point for one job
  github/                 job metadata and log fetching
  labeler/                preprocess, prompt, call, parse, store
  dataset/                loading and split filtering
  evaluation/             confusion matrix and per class metrics

tools/
  run_eval.py             evaluate a split, print the report
  make_split.py           verify the committed split
  smoke_test.py           manual end to end check
```

## Status

Phase 1, the offline labeler, is complete and measured. Remaining work is
tracked in `EXPERIMENTS.md`:

* Quantify and remove the docstring leakage in the dataset (E4)
* Prompt v2 targeting the `genuine_regression` fallback
* Preprocessing ablation (E1) and model comparison (E2)
* Unit tests for the pure functions in `metrics.py` and `log_preprocessor.py`

Phase 2 will trigger the pipeline automatically on CI failure and add evidence
gathering: when confidence is low or the flaky against regression boundary is in
play, fetch the job's re run history and the commit diff, then re classify. Re
run history is the stronger signal of the two, since the same commit passing on
an earlier attempt is close to conclusive evidence of flakiness.
