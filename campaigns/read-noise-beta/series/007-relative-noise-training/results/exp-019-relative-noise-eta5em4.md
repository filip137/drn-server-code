---
experiment: "exp-019"
evidence: "validated-local"
summary: "At eta=5e-4, fixed-final-epoch ours minus legacy is -0.04/+0.40/+6.46 pp for Conv1/2/3; nine seed-0 cases validated."
verdicts: {"H-011": "inconclusive"}
---
# Result: exp019 — Relative endpoint noise eta=5e-4

## Evidence and coverage

Reviewed September 28, 2026 against the [frozen experiment](../experiments/exp-019-relative-noise-eta5em4.md).
All nine declared cases completed once: Conv1 10 epochs and Conv2/3 30 uninterrupted
epochs, baseline/ours/legacy, seed 0. No failed, replaced, excluded or resumed
training cases. Nine CPU semantic smokes preceded production; no remote canary.

The exact remote study was collected with rsync into
`results/conv123-relative-noise-eta5em4-jeanzay-20260928-v1/collected/`, retaining remote
evidence. The study's `analysis/analyze.py` passed for 9/9 canonical bundles using
`experiments.reporting.validate_run`: hashes, frozen resolved configs, finite ordered
epoch metrics, complete checkpoint indices 0..N, 438 model/optimizer checkpoints,
and matching initializer/initial-parameter identities within each depth. All 210
epoch records are present. Runtime records confirm zero official-test evaluations
and zero official-test examples; all measurements are ordinary-MNIST validation.

Source commit: `d9f581fbef9ba40401a073f1e80c9c85ffdb1d75`; source archive SHA256:
`1fb67f7fa24bd9758b1a90c61bb04ddc687998916e6046be7405531b0d761e64`.
Per-arm config/input hashes and local bundle paths are in `analysis/summary.json`.
All nine GPU receipts identify V100s and the authorized accounts. Fresh Slurm
accounting confirms jobs 272329–272337 COMPLETED, exit 0:0, with none queued/running.
Total allocation time was 73,608 GPU-seconds (**20.4467 GPU-hours**), including
startup/teardown; launcher receipts total 73,584 seconds. Charge the larger allocation
total, leaving 9.5533 hours under the original 30-hour cap, with no new work implied.
Accounting, launcher and collection receipts are in the study's `analysis/` directory.

An analysis-only schema error was repaired: the production dataset policy field is
`official_test_policy`; `configured_official_test_policy` exists only for diagnostic
overrides. The initial validation failure is preserved as
`analysis/validation-schema-failure.json`. The corrected analyzer retained the strict
disabled-test requirement and passed; no scientific config, runner or result changed.

## Measurements

Validation accuracy (%), at the predeclared fixed final epoch:

| Model | Epoch | Baseline | Ours | Legacy | Ours − legacy (pp) | Ours − baseline (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Conv1 | 10 | 95.84 | 96.08 | 96.12 | -0.04 | +0.24 |
| Conv2 | 30 | 92.98 | 96.06 | 95.66 | +0.40 | +3.08 |
| Conv3 | 30 | 83.18 | 88.76 | 82.30 | +6.46 | +5.58 |

Secondary epoch-10 accuracy (baseline/ours/legacy): Conv2 90.32/95.38/94.80%,
Conv3 81.18/86.94/86.08%. Thus epoch-10 ours-minus-legacy is +0.58 and +0.86 pp.
Best-epoch results are retained only as diagnostics in `analysis/accuracy.csv` and
`analysis/report.md`; they do not replace the fixed-epoch comparisons.

[Fixed-epoch JPG](../../../../../results/conv123-relative-noise-eta5em4-jeanzay-20260928-v1/analysis/final_accuracy.jpg).

## Interpretation

At this added eta and seed, the declared positive-gap criterion is met for Conv2
and Conv3 at their final epochs, but not Conv1. Ours exceeds baseline in all three
tested cells. Conv3's large final gap includes legacy declining from 86.08% at
epoch 10 to 82.30% at epoch 30; the result does not identify the mechanism of that
decline. Low accuracy is valid scientific evidence, not an execution failure.

H-011 concerns epoch-10 transfer across the original complete grid. This added cell
supports transfer at epoch 10 for Conv2 and Conv3, but has a negative Conv1 pair.
The overall H-011 verdict here is inconclusive: this single added level neither
establishes all-architecture transfer nor overturns the original grid's findings.
Do not reinterpret epoch-30 gaps as direct tests of its epoch-10 claim.

These are exploratory single-seed finite-iteration EqProp validation measurements,
not significance, reproducibility, an established noise optimum, or official-test
paper accuracy. Historical exp015 epoch-30 controls reset Adam/RNG after epoch 10;
they cannot be pooled with these uninterrupted runs to isolate eta effects.

## Decision

Close exp019 with all authorized coverage validated and reviewed. Retain the
negative Conv1 pair and positive deeper-network pairs. No retuning or additional
run is authorized by this result; locating an advantage optimum remains unresolved.
