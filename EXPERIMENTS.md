# Experiments

A running log of decisions made against measurement rather than intuition.
Each entry states the question, the setup, the numbers, and what was decided.
Hypotheses are written before the run where possible, so a wrong one stays on
the record.

## Conventions

* All experiments run on the 27 example dev_eval split. The 10 final_check IDs
  are a holdout and stay untouched until Phase 1 is closed.
* Metrics are reported per class with support counts, never as a single
  accuracy figure. The split is imbalanced (10 genuine_regression against 3
  infra), so an aggregate number is dominated by the largest class.
* Baselines for comparison: random across 4 classes (0.25) and majority class
  (0.37, always answering genuine_regression).
* Temperature is fixed at 0.0 throughout.
* Every label is persisted in `data/labels.jsonl`, keyed on example_id,
  prompt_version, model, and preprocessing_variant. Changing any of those four
  forces a fresh API call rather than reusing a stale answer.
* n = 27 is small. Every number here carries a wide confidence interval, and a
  difference of one or two examples is not evidence of anything.

## E0: Baseline

**Question:** Can a single zero shot LLM call classify a CI failure from its
log alone, and does it beat a trivial classifier?

**Hypothesis (written before the run):** Yes on the trivial baselines. The
flaky_test vs transient boundary was expected to be the main source of error,
since both are non deterministic failures.

**Setup**

| Setting | Value |
|:-|:-|
| Split | dev_eval (27 examples) |
| Model | claude-sonnet-4-6 |
| Prompt version | v1 (zero shot) |
| Preprocessing variant | strip_none |
| Temperature | 0.0 |
| Window | head 40, tail 60 |

Reproduce with `python -m tools.run_eval` from the project root.

**Results**

```
Class                Precision    Recall        F1   Support
flaky_test                1.00      0.86      0.92         7
genuine_regression        0.71      1.00      0.83        10
infra                     1.00      0.67      0.80         3
transient                 1.00      0.71      0.83         7
Overall accuracy          0.85                            27

CONFUSION MATRIX
(rows = true label, columns = predicted)

true \ pred          flaky_tes genuine_r     infra transient
flaky_test                   6         1         0         0
genuine_regression           0        10         0         0
infra                        0         1         2         0
transient                    0         2         0         5

BASELINES
Random (4 classes)        0.25
Majority class            0.37   (genuine_regression)
```

**Finding 1: every error is the same error.**

There were 4 misclassifications, and all 4 were predicted as
genuine_regression. No other pair of categories was ever confused. The
flaky_test vs transient boundary, expected to be the hard one, was completely
clean at 0 errors in both directions. The hypothesis was wrong about where the
errors would come from.

This shows up in the metrics as the only asymmetric pair: genuine_regression
has recall 1.00 and precision 0.71. It never misses a real regression, and it
over claims on everything else.

**Finding 2: genuine_regression is the model's fallback answer.**

Reading `prompts/v1.txt` against this result, the disambiguation section has
explicit rules for flaky_test vs transient and for transient vs infra, but
nothing that pushes back against genuine_regression. Its definition is also the
only one that does not require positive evidence of anything unusual, so it
wins whenever the log does not clearly announce another category.

Operationally this is the expensive direction of error: over calling
genuine_regression pages an engineer to debug a network timeout.

**Caveat: this number is almost certainly inflated.**

The fault injected tests carry self describing docstrings and names, for
example `test_flaky_currency_rounding` with a docstring reading "Intentionally
flaky: depends on a random value rather than a deterministic condition".
pytest prints test source on failure, so that text reaches the prompt.
Inspected rationales cite it directly. The model is reading an English label,
not diagnosing a failure.

The clean flaky_test vs transient separation is consistent with that reading,
and `transient_001` and `transient_002` (both missed) may be the cases where
the docstring did not announce the answer. E4 quantifies the size of this
effect. Until then, 0.85 should not be quoted as a diagnostic accuracy.

**Decision:** Accepted as the baseline for all later comparisons. No changes
made to the prompt or preprocessing on the strength of it.

**Next actions**

1. E4 first, to establish how much of this is leakage. Prompt work before that
   would be uninterpretable, since any gain could be a change in how much
   annotation the model happens to read.
2. Prompt v2 targeting the genuine_regression fallback, measured against E0.

## E1: Preprocessing ablation

**Question:** Does removing the runner env block and pip install output change
classification quality?

**Hypothesis (written before the run):** No gain on flaky_test, transient, or
genuine_regression. Possible loss on infra, where environment variables such as
`pythonLocation` and `LD_LIBRARY_PATH` are the diagnostic evidence.

**Setup:** `strip_env_block` and `strip_pip_output` each true or false, giving 4
variants. Model (claude-sonnet-4-6), prompt version (v1), and window sizes fixed
at E0 values. Structural strips (timestamps, ANSI escapes, group markers) stay
on throughout. 81 additional API calls across the three new variants.

**Results**

| Variant | flaky P/R | genuine P/R | infra P/R | transient P/R | Accuracy |
|:-|:-|:-|:-|:-|:-|
| strip_none (E0) | 1.00 / 0.86 | 0.71 / 1.00 | 1.00 / 0.67 | 1.00 / 0.71 | 0.85 |
| strip_env | 1.00 / 0.86 | 0.71 / 1.00 | 1.00 / 0.67 | 1.00 / 0.71 | 0.85 |
| strip_pip | 1.00 / 0.86 | 0.71 / 1.00 | 1.00 / 0.67 | 1.00 / 0.71 | 0.85 |
| strip_env_pip | 1.00 / 0.86 | 0.77 / 1.00 | 1.00 / 1.00 | 1.00 / 0.71 | 0.89 |

**Finding: the four variants differ by exactly one example.**

Diffing the confusion matrices cell by cell, `strip_env` and `strip_pip` are
identical to `strip_none` in every position. `strip_env_pip` differs in two
cells, which is one example (`infra_004`) moving from genuine_regression to
infra.

That single example accounts for the entire difference between the reported
accuracies. 23 of 27 correct becomes 24 of 27. It also produces the
genuine_regression precision change, since predictions in that column drop from
14 to 13, so 10/14 becomes 10/13.

**The one example that moved was not a borderline case.**

The obvious explanation was that `infra_004` sat near a decision boundary and a
shorter prompt tipped it. The stored confidences rule that out:

| Variant | Label | Confidence |
|:-|:-|:-|
| strip_none | genuine_regression | 0.85 |
| strip_env | genuine_regression | 0.85 |
| strip_pip | genuine_regression | 0.85 |
| strip_env_pip | infra | 0.92 |

Three confident wrong answers, then a more confident right one. Truncation was
also ruled out: the log is 28 lines after stripping against a window threshold
of 100, so no variant truncated anything.

**Reading the rationales explains it, and the cause is the prompt.**

All four runs diagnosed the same root cause: `requirements.txt` pins
`starlette==0.27.0` while `fastapi 0.115.0` requires `>=0.37.2`. The model never
disagreed about what happened. It disagreed about which category that belongs
to.

Three reasoned that a version conflict committed to the repository is
deterministic and reproduces on every run, which is genuine_regression. One
reasoned that the failure occurred during dependency installation before any
tests ran, which is infra.

Both readings follow from the v1 prompt, because its category definitions
overlap for this case. `infra` lists dependency resolution failure explicitly,
and `genuine_regression` is defined by determinism, and a bad version pin
satisfies both.

**The deeper problem: v1 used determinism as the discriminator, and it does not
discriminate.**

`data/dataset/LABELING_RUBRIC.md` states that infra is also deterministic, and
gives the actual discriminator in one sentence: no application code is involved.
The rubric's decision procedure asks whether the application is at fault first,
and only then asks whether a re-run would pass.

The v1 prompt inverted this. Its only determinism rule reads "if a failure is
deterministic and reproducible, it is genuine_regression", which points the
opposite way from the rubric for any deterministic non-application failure.

The rubric header states that its class definitions are reused verbatim in the
classifier prompt. They were not. That drift is the bug.

**Corrected conclusion on the interaction.**

The earlier reading, that neither flag mattered alone but both together produced
a real interaction, does not hold. Preprocessing did not fix anything. The
ambiguity is present in all four variants, and one example happened to resolve
differently under a shorter prompt. The fix belongs in the prompt, not in
preprocessing, and is tracked as prompt v2.

**Multiple comparisons caveat.**

Four variants were evaluated on the same 27 examples and the best was selected.
With n this small, selecting the maximum across several variants tends to select
noise rather than signal. This is the same reason the final_check split exists
and is not being used for tuning.

**Decision: adopt `strip_env_pip`, for the token cost, not the accuracy.**

The defensible claim is that no variant produced a difference distinguishable
from noise, so the choice comes down to cost. Stripping removes roughly seven
lines of env block plus the pip output line from a 28 line log, and accuracy did
not degrade, which is the condition the flag was gated on.

The claim being explicitly avoided: that stripping improved accuracy from 0.85
to 0.89. That is one example, it moved for a reason unrelated to preprocessing,
and reporting it as an improvement would be the kind of overclaim the support
counts exist to prevent.

**Caveat carried forward:** `pip_output_patterns` enumerates known pip line
shapes rather than matching structurally, so with the flag on, some pip output
still survives. Any measured effect is a lower bound on what full pip stripping
would do.

**Constraint:** `strip_env_pip` is now the production setting as well as the
evaluation setting. A flag that differs between the two is train/serve skew.

## E2: Model comparison (not started)

**Question:** Which model gives the best accuracy for the cost, as a labeler?

**Setup:** Haiku, Sonnet, and Opus. Prompt version and preprocessing fixed at
whatever E1 settles on. Full dev_eval run for each.

| Model | flaky P/R | genuine P/R | infra P/R | transient P/R | Accuracy | Cost |
|:-|:-|:-|:-|:-|:-|:-|
| claude-sonnet-4-6 | 1.00 / 0.86 | 0.71 / 1.00 | 1.00 / 0.67 | 1.00 / 0.71 | 0.85 | |
| Haiku | | | | | | |
| Opus | | | | | | |

**Decision:** (pending)

**Note on cost:** the instruction block is identical across all 27 calls, so
prompt caching would cut input cost substantially at larger scale. At n = 27
the saving is pennies and is not worth implementing, but it is the obvious
lever if this ran per commit across many repositories.

## E3: Keyword anchored windowing (not started)

**Question:** Does anchoring the window on pytest session markers beat blind
head plus tail truncation?

**Setup:** v1 keeps `head_lines` and `tail_lines` as the deterministic
baseline. The variant anchors on the pytest session start and short test
summary markers when both are present, and falls back to head plus tail when
they are absent.

**Why the fallback is required:** infra failures die during environment setup,
before pytest ever runs, so the markers do not exist in those logs at all. An
anchor only strategy would return an empty window for exactly the class with
the lowest support.

**Decision:** (pending)

## E4: Label leakage quantification (not started, blocks E0 interpretation)

**Question:** How much of the E0 accuracy comes from the model reading a self
describing docstring rather than diagnosing the failure?

**Background:** the fault injection wrote tests with names and docstrings that
state their own category in plain English. pytest prints test source on
failure, so that text lands in the prompt. Two inspected rationales cite the
docstring as their primary evidence, both at 0.99 confidence.

**Setup:** run dev_eval twice, once as E0 and once with docstrings and comments
stripped from the log before prompting. The difference in per class recall is
the inflation.

| Variant | flaky P/R | genuine P/R | infra P/R | transient P/R | Accuracy |
|:-|:-|:-|:-|:-|:-|
| E0 (docstrings present) | 1.00 / 0.86 | 0.71 / 1.00 | 1.00 / 0.67 | 1.00 / 0.71 | 0.85 |
| Docstrings stripped | | | | | |

**Expected outcome:** a drop, concentrated in flaky_test, since that is where
the docstrings are most explicit.

**Follow up regardless of the number:** rewrite the injected tests with neutral
names and docstrings, regenerate the logs through CI, and re establish the
baseline. Stripping docstrings at preprocessing time is a workaround, not a
fix, because real tests carry legitimate docstrings that are genuine evidence.

**Decision:** (pending)

## Template for new entries

```
## E<n>: <title>

**Question:**

**Hypothesis:** (written before the run)

**Setup:** model, prompt version, preprocessing variant, what varies, what is
held fixed

**Results:** per class table with support

**Finding:**

**Decision:**

**Caveats:**
```