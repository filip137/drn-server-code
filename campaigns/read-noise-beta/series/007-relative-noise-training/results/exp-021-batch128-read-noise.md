---
experiment: "exp-021"
evidence: "validated-local"
summary: "Partial batch-size review: Conv3 clean accuracy is preserved at batch128; eta5e-4 accuracy rises by 4.88–6.20 pp across schemes, with one lower-noise case outstanding."
verdicts: {}
---
# Exp021 — Batch size and read-noise sensitivity

## Evidence and scope

Partial review recorded September 29, 2026 at Filip's request, against the
[frozen study](../experiments/exp-021-batch128-read-noise.md). The local collector
validated 44/45 cases at 11:46:58 Paris with no invalid included cases. All Conv1
and Conv2 cases and 14/15 Conv3 cases are included. Conv3 legacy EP eta1e-5 is
outstanding; its interrupted attempt and queued full-state continuation are
preserved in the experiment handoff. No intermediate accuracy substitutes for
that missing final result. The study remains active and is not closed out.

Current evidence root: `results/conv123-read-noise-batch128-jeanzay-20260929-v1/`.
`analysis/summary.json` records final accuracies, validation state, input/source
identities, cohort provenance and bundle paths. Validation used
`launch/jean-zay/collect.py --allow-partial`: canonical bundles, fixed epoch
coverage, finite model/Adam state, checkpoint hashes, within-depth matching,
full RNG state, and zero official-test evaluations. Failed startup and recovery
attempts remain preserved; they are not extra seeds. Frozen numerical source
identity: `eeea0b654abdcac01745625993b366b8da955865fba2e617c0d1e3a99f5d301c`.

These comparisons use seed0, ordinary-MNIST 55k/5k training/validation, the same
declared learning rates, beta, initialization, precision and iteration settings,
and fixed horizons of 10/30/30 epochs for Conv1/2/3. Read noise is independent
Gaussian endpoint noise with sigma = eta × absolute clean voltage; validation
is clean. All percentages below are final-epoch validation accuracy, not selected
best epochs or official-test accuracy. This is a retrospective comparison of
historical runs, not a newly randomized experiment isolating batch size.

Historical batch16 evidence, already collected and reviewed:

- [exp016](exp-016-noiseless-three-seed.md): uninterrupted clean Conv2/3,
  using seed0 rather than the three-seed mean. Roots:
  `results/conv23-noiseless-three-seed-20260927-v1/` and
  `results/conv23-noiseless-baseline-three-seed-20260927-v1/`.
- [exp017](exp-017-baseline-relative-noise.md): uninterrupted baseline noisy
  Conv2/3; `results/conv23-baseline-relative-noise-20260927-v1/`.
- [exp015](exp-015-epoch30-continuation.md): ours/legacy eta1e-5 and1e-4,
  with Adam and data/noise RNG reset after epoch10; roots
  `results/conv23-relative-noise-epoch30-20260926-v1/` and
  `results/conv23-relative-noise-epoch30-low-noise-20260926-v1/`.
- [exp019](exp-019-relative-noise-eta5em4.md): uninterrupted eta5e-4 for all
  depths and schemes, on V100; root
  `results/conv123-relative-noise-eta5em4-jeanzay-20260928-v1/`.

Per-case values come from the respective `analysis/accuracy.csv` files and the
exp019 `analysis/summary.json`; current values come from exp021's validated summary.

## Conv3 accuracy: batch16 → batch128

All cells are EP, seed0, epoch30. Asterisks mark historical optimizer/RNG resets.

| Relative noise eta | Baseline (%) | Ours (%) | Legacy (%) |
|---|---:|---:|---:|
| 0 | 97.56 → 97.50 | 98.76 → 98.66 | 98.70 → 98.78 |
| 1e-5 | 93.66 → 95.30 | 96.10* → 97.30 | 96.72* → pending |
| 1e-4 | 85.46 → 93.46 | 95.80* → 95.52 | 89.00* → 94.66 |
| 5e-4 | 83.18 → 89.38 | 88.76 → 93.64 | 82.30 → 88.28 |

Clean accuracy changes are small: −0.06/−0.10/+0.08 percentage points for
baseline/ours/legacy. At eta5e-4, increases are +6.20/+4.88/+5.98 pp. The
clean-to-noisy accuracy drop decreases from 14.38 to 8.12 pp for baseline,
10.00 to 5.02 pp for ours, and 16.40 to 10.50 pp for legacy. Thus higher noisy
accuracy is accompanied by nearly unchanged clean accuracy in this comparison.

Ours-minus-legacy at eta5e-4 is +6.46 pp with batch16 and +5.36 pp with batch128.
The larger batch improves both schemes but does not enlarge ours' relative
advantage at this noise level. At eta1e-4 the gap changes from +6.80 to +0.86 pp;
that comparison is confounded by the batch16 optimizer/RNG reset. At clean eta0,
the small +0.06 pp batch16 gap becomes −0.12 pp at batch128.

## Strong-noise batch effect across depth

Batch128 minus batch16 final EP accuracy at eta5e-4, in percentage points.
Both batch sizes trained uninterrupted for these included cells.

| Architecture | Final epoch | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| Conv1 | 10 | +0.06 | +0.12 | −0.02 |
| Conv2 | 30 | +1.46 | +0.58 | +0.46 |
| Conv3 | 30 | +6.20 | +4.88 | +5.98 |

The observed benefit at this noise level is concentrated in the deeper network:
Conv1 changes little, Conv2 improves modestly, and Conv3 improves substantially.
The differing Conv1 epoch horizon and single seed limit generalization across depth.

## Runtime and interpretation

Matched V10032 EP timing at eta5e-4 shows approximately 2.8–2.9× faster Conv2
epochs and 2.1–2.2× faster Conv3 epochs with batch128. Conv3 takes roughly
3.9–4.0 minutes per epoch versus 8.6–8.7 minutes for batch16. These are measured
intervals between epoch records, including intervening validation/checkpoint
overhead but excluding startup and queue wait. Timing provenance and method
are in `analysis/batch16-timing-comparison.json` under the exp021 root. Recovery
and queue delays remain real study costs and are not included in those speedups.

At fixed epochs and unchanged learning rates, batch128 provides similar clean
Conv3 accuracy and better accuracy at eta5e-4 with shorter training time in this
seed. This supports larger batches as a useful operating choice for the tested
read-noise regime. It does not establish a universal batch-size advantage: ours
at eta1e-4 is slightly lower, some historical optimizer histories differ, and
only one seed and two batch sizes are compared.

There are 430 optimizer updates per epoch at batch128 versus 3438 at batch16.
Batch averaging and fewer noisy parameter updates are possible explanations,
but the evidence does not distinguish them from changed optimization dynamics.
No direct gradient-noise variance or cosine-similarity measurement was made in
this comparison. Do not infer an eightfold noise reduction or a demonstrated
mechanism from accuracy alone. Historical source/runtime snapshots and some
hardware placements also differ; no cross-snapshot numerical-equivalence audit
is claimed. The uninterrupted V100 eta5e-4 comparison avoids the known Adam-reset
and GPU-model differences but still compares historical executions.

## Decision and remaining work

Retain this as a partial, descriptive batch-size result. Finish and validate the
remaining Conv3 legacy eta1e-5 continuation, then reconcile complete coverage
and all attempt costs before whole-study review. This epoch30 comparison does
not assign a new verdict to H-011's original epoch10 claim. No extra seeds,
batch sizes, tuning, mechanism diagnostics or new simulations are authorized
by this report.
