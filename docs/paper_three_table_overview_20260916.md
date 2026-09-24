# Working overview and experimental-section draft

Snapshot: 2026-09-16. This overview follows Filip's proposed organization:
wide-range BPTT/EqProp, bounded-weight BPTT, and EqProp read-noise robustness.
Table numbering below is the proposed manuscript numbering, which differs
from the older collection directory names.

**Read-noise review note, September 23: beta must be selected first.** Before
finalizing the paper's read-noise robustness comparison, choose and freeze a
defensible beta rule and the corresponding scheme-specific values. Existing
noise results remain evidence at their recorded betas; their inclusion and
matching clean controls must be reassessed after selection. See the
[current beta objective](current_state.md#current-focus-finding-the-right-beta).

All accuracies here are **ordinary-MNIST validation accuracies**, not official
test results. The official 10,000-example test split remains unread. Tables
1 and 2 report mean best-checkpoint validation accuracy ± sample standard
deviation across model/shuffle seeds 0, 1, and 2. Table 3 reports the same
best-checkpoint metric for seed 0, without an uncertainty estimate. Final
noise accuracies are preserved separately below.

For this working presentation, Table 2 uses the original shared-T/K BPTT
results. These provide complete three-seed coverage. The later longer-T
baseline repeats are displayed separately; their results are not mixed into
the shared-T/K table. This presentation does not amend the current revised
training ledger or seal the final paper inclusion set.

| Proposed table | Collected training coverage | Seed coverage | Outstanding for the proposed coverage |
|---|---|---|---|
| 1: wide BPTT versus EqProp, all schemes | 54/54 | Three in every cell | Qualification/inclusion decisions and official-test evaluation |
| 2: bounded BPTT, all schemes, original shared T/K | 81/81 | Three in every cell | Final choice of baseline contract, reuse seal and official-test evaluation |
| 3: wide EqProp with read noise, all schemes | 30/45 in this September 16 snapshot | One per completed noise cell | First select and freeze beta; then reconcile matching clean/noisy coverage and complete the official-test handoff |

The 45-cell count in the last row describes extending the completed
30-cell ours/legacy sweep to all three schemes; it is not a newly authorized
launch plan or compute reservation. Additional seeds would measure
variability. No additional training is needed merely to populate Tables 1
and 2 under the displayed settings. A subsequent scientific-contract change
could require affected runs to be repeated.

All three schemes use perfect-diode Conv networks and Adam. Baseline, ours
and legacy have voltage/current amplification `(1,1)`, `(4,1)` and
`(4,0.25)`, respectively. Biases are fixed at zero. Training lasts 10 epochs
for Conv1 and 30 for Conv2/Conv3, with batch size 16. The common
train/validation split has 55,000/5,000 examples and split seed 0.
The displayed main tables use T/K = 4/4, 6/6 and 8/8. T is free/inference
relaxation; K is the additional BPTT unroll or the length of each centered
EqProp nudged phase.

Table 1 compares clean BPTT and centered EqProp under wide conductance bounds
`[0,100]`. Values are percentages, mean ± sample SD, n = 3.

| Architecture | Algorithm | Baseline | Ours | Legacy |
|---|---|---:|---:|---:|
| Conv1 | BPTT | 96.27 ± 0.11 | 96.49 ± 0.08 | 96.54 ± 0.05 |
| Conv1 | EqProp | 96.28 ± 0.09 | 96.45 ± 0.08 | 96.53 ± 0.06 |
| Conv2 | BPTT | 97.34 ± 0.02 | 98.19 ± 0.08 | 98.30 ± 0.04 |
| Conv2 | EqProp | 97.29 ± 0.08 | 98.15 ± 0.06 | 98.34 ± 0.04 |
| Conv3 | BPTT | 97.77 ± 0.03 | 98.56 ± 0.10 | 98.89 ± 0.06 |
| Conv3 | EqProp | 97.69 ± 0.08 | 98.60 ± 0.11 | 98.91 ± 0.07 |

Both amplified schemes exceed baseline in each architecture/algorithm mean;
legacy has the highest mean throughout. The largest absolute BPTT–EqProp
difference between corresponding three-seed means is 0.0733 percentage
points. Similar training accuracy does not establish gradient equality or
full equilibration. In particular, wide Conv3 baseline beta qualification
remains unresolved, and Conv3 ours retains its inherited gradient exception
at injected beta 3. The existing training results remain available with
those limitations.

Table 2 compares bounded-weight BPTT at shared T/K. The lower conductance
bound is `Gmin = 1e-5`; the upper bound varies as shown. All cells have three
seeds. These runs use `bounded_uniform` initialization, whereas the wide
runs use `kaiming_uniform`, so the wide-to-bounded comparison changes both
the range and initialization contract.

| Architecture | Gmax | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| Conv1 | 0.0001 | 89.85 ± 0.16 | 93.04 ± 0.32 | 94.94 ± 0.09 |
| Conv1 | 0.0005 | 94.58 ± 0.12 | 95.17 ± 0.22 | 96.36 ± 0.10 |
| Conv1 | 0.001 | 94.95 ± 0.23 | 95.17 ± 0.20 | 96.42 ± 0.08 |
| Conv2 | 0.0001 | 85.45 ± 0.13 | 90.19 ± 0.17 | 95.63 ± 0.21 |
| Conv2 | 0.0005 | 92.04 ± 0.46 | 96.56 ± 0.12 | 97.68 ± 0.03 |
| Conv2 | 0.001 | 93.46 ± 0.10 | 97.13 ± 0.06 | 97.87 ± 0.08 |
| Conv3 | 0.0001 | 77.60 ± 0.05 | 85.13 ± 0.59 | 93.92 ± 0.97 |
| Conv3 | 0.0005 | 87.87 ± 0.43 | 94.62 ± 0.21 | 97.83 ± 0.14 |
| Conv3 | 0.001 | 89.80 ± 0.22 | 95.73 ± 0.06 | 98.01 ± 0.18 |

Legacy retains the highest mean accuracy at every tested ceiling. The
separation is largest in deeper networks under the tightest bound: Conv3
reaches 77.60%, 85.13% and 93.92% for baseline, ours and legacy. These
comparisons retain scheme-specific learning-rate vectors; they do not
isolate amplification from every other optimization choice.

The additional bounded baseline BPTT repeats below use longer free
relaxation. Conv1 was unchanged. Conv2 has all three seeds at T12/K6;
Conv3 has seeds 0 and 1 at T24/K8, with seed 2 missing at each ceiling.
Incomplete groups show individual outcomes, not a three-seed aggregate.

| Architecture | T/K | Gmax | Best validation (%) | Seeds |
|---|---:|---:|---:|---:|
| Conv2 | 12/6 | 0.0001 | 85.39 ± 0.14 | 3 |
| Conv2 | 12/6 | 0.0005 | 92.04 ± 0.46 | 3 |
| Conv2 | 12/6 | 0.001 | 93.55 ± 0.06 | 3 |
| Conv3 | 24/8 | 0.0001 | seed 0: 77.68, seed 1: 77.50 | 2 |
| Conv3 | 24/8 | 0.0005 | seed 0: 87.12, seed 1: 87.36 | 2 |
| Conv3 | 24/8 | 0.001 | seed 0: 89.38, seed 1: 89.58 | 2 |

Table 3 covers EqProp training with independent Gaussian endpoint-voltage
read noise, at wide conductance bounds `[0,100]`. Noise affects the local
gradient readout; inputs, relaxation dynamics and validation remain
noiseless. Sigma is in simulator voltage units. Values below are
best-checkpoint validation percentages for model/shuffle seed 0 and noise
seed 2026081601. The clean column reuses the earlier seed-0 run at the same
beta; it is not a newly executed sigma-zero control. Baseline noise runs and
their final control contract are missing; existing clean baseline results
are already represented in Table 1.

| Architecture | Scheme | Clean seed 0 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 5e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Conv1 | baseline | pending control | — | — | — | — | — |
| Conv1 | ours | 96.42 | 96.44 | 96.44 | 96.40 | 96.44 | 96.50 |
| Conv1 | legacy | 96.48 | 96.52 | 96.50 | 96.56 | 96.52 | 96.48 |
| Conv2 | baseline | pending control | — | — | — | — | — |
| Conv2 | ours | 98.10 | 98.12 | 98.08 | 98.02 | 97.88 | 97.66 |
| Conv2 | legacy | 98.30 | 97.68 | 97.40 | 96.64 | 95.98 | 95.72 |
| Conv3 | baseline | pending control | — | — | — | — | — |
| Conv3 | ours | 98.72 | 98.38 | 97.88 | 97.62 | 97.12 | 96.78 |
| Conv3 | legacy | 98.84 | 96.12 | 93.42 | 84.66 | 84.92 | 76.36 |

At sigma `5e-4`, Conv2 loses 0.44 percentage points for ours versus 2.58
for legacy relative to their clean best checkpoints. Conv3 loses 1.94
versus 22.48 points. Conv1 has little observed change over this grid. All
30 noisy trainings completed with finite checkpoints; severe accuracy
degradation is not the same as numerical divergence.

The injected betas are fixed across noise levels but differ between schemes:
Conv1 ours/legacy = 30/3; Conv2 = 10/0.03; Conv3 = 3/0.001. Each Conv1/2
within-sigma pair shares a host and audited noise stream. Every Conv3 pair
crosses execution-environment/noise-stream groups, so Conv3 provides a
descriptive robustness comparison rather than a controlled attribution to
amplification alone. Clean references also use an earlier environment.
Only one training/noise seed is measured. These limits and the inherited
Conv3-ours beta exception must accompany the interpretation.

For training-stability analysis, the corresponding final-epoch noise
accuracies are:

| Architecture | Scheme | Clean seed 0 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 5e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Conv1 | ours | 96.34 | 96.34 | 96.30 | 96.34 | 96.30 | 96.28 |
| Conv1 | legacy | 96.44 | 96.40 | 96.40 | 96.40 | 96.36 | 96.38 |
| Conv2 | ours | 98.02 | 98.08 | 98.08 | 98.00 | 97.80 | 97.60 |
| Conv2 | legacy | 98.06 | 97.62 | 97.38 | 96.52 | 95.94 | 95.72 |
| Conv3 | ours | 98.64 | 98.26 | 97.82 | 97.62 | 96.82 | 96.60 |
| Conv3 | legacy | 98.78 | 95.68 | 93.42 | 83.08 | 82.34 | 75.44 |

## Initial experimental-section prose

The paragraphs below are a selection-stage draft. They intentionally identify
the displayed metrics as validation results. Their numerical claims must be
recomputed from sealed official-test evaluations before becoming the final
paper accuracy section.

We evaluate three convolutional dissipative resistive networks, Conv1,
Conv2 and Conv3, with one, two and three hidden convolutional layers. Hidden
channel counts are 64, 64/128 and 64/128/256, respectively. All convolutions
use 3-by-3 kernels and padding one; strides are 2, 2/2 and 2/2/1, with no
pooling. The networks use perfect-diode nonlinearities and 20 paired outputs
for ten-class classification. We compare the unamplified baseline
`(Av,Ai)=(1,1)`, our amplification scheme `(4,1)`, and the legacy scheme
`(4,0.25)` on ordinary MNIST, using a fixed 55,000/5,000 train/validation
split. All reported training uses Adam, batch size 16 and zero biases.
Conv1 trains for 10 epochs and Conv2/Conv3 for 30. The clean comparisons use
three model and shuffle seeds, with learning rates fixed separately for
each architecture, scheme and weight contract. We summarize the maximum
validation accuracy within each training budget and report its mean and
sample standard deviation across seeds. Official-test evaluation remains
reserved for the final frozen protocol and selected checkpoints.

First, we compare BPTT and centered EqProp under wide conductance bounds
`[0,100]` (Table 1). Each algorithm pair shares its learning-rate vector,
initialization, training order and iteration budget: T/K = 4/4, 6/6 and 8/8
for Conv1/Conv2/Conv3. Both amplified schemes improve the observed mean
validation accuracy over the baseline at every depth, and legacy has the
highest mean in each comparison. BPTT and EqProp differ by less than 0.08
percentage points in their corresponding three-seed means. This supports
similar empirical training performance under the selected finite-step
contracts; it does not establish gradient equivalence, and the remaining
EqProp qualification exceptions are reported separately.

Second, we examine BPTT with conductances restricted to
`[1e-5,Gmax]`, where `Gmax` is `1e-4`, `5e-4` or `1e-3` (Table 2).
The comparison keeps the original iteration budget shared across schemes
and uses bounded-uniform initialization. Legacy has the highest mean
validation accuracy at every tested ceiling. At the tightest bound, Conv3
achieves 77.60%, 85.13% and 93.92% for baseline, ours and legacy,
respectively. Increasing the ceiling improves all three schemes, with
legacy showing the smallest accuracy change. The later baseline repeats
with longer free relaxation are reported separately because they change
the computation budget.

Third, we test the effect of endpoint-voltage read noise during EqProp
training (Table 3). Independent Gaussian perturbations with standard
deviation between `1e-5` and `5e-4` are applied to the endpoint voltages
used to estimate local gradients; relaxation and evaluation remain
noiseless. The completed one-seed sweep covers ours and legacy at all
five noise levels. Conv1 changes little, while ours retains more accuracy
than legacy in Conv2 and Conv3 at every measured level. At sigma `5e-4`,
the best Conv3 checkpoints reach 96.78% for ours and 76.36% for legacy,
compared with clean references of 98.72% and 98.84%. This is evidence of
different observed noise sensitivity under the selected scheme/beta
contracts. Baseline coverage remains incomplete, and the single seed,
scheme-dependent betas and unmatched Conv3 execution environments limit
causal and uncertainty claims.

## Remaining work and supporting evidence

- Decide the final Table 2 baseline contract. The original shared-T/K
  BPTT surface is complete; the longer-T alternative needs only three
  Conv3 seed-2 BPTT runs to reach three seeds everywhere. Bounded EqProp
  completion is outside the proposed main Table 2.
- Complete the baseline portion of Table 3 after its clean beta/control
  contract is resolved. At the current five noise values this adds 15
  noisy trainings for one seed. A changed beta requires a matching clean
  reference. More seeds are needed to estimate noise-study variability.
- Preserve the observed Conv3 noise results, but use matched execution
  environments and verified noise draws for a stronger causal comparison.
  Resolve or explicitly retain the scientific EqProp exceptions according
  to the claims the paper makes.
- Audit the final inclusion set, seal the maximum-validation checkpoints
  and evaluate each eligible checkpoint on the official test set exactly
  once. The displayed validation values are not substitutes for that step.
- Update manuscript tables and any retained legacy mechanism figures from
  the corrected physical-KCL sources. The existing experimental section
  still describes the older single-seed organization.

Additional evidence is available for a later mechanism section or appendix:
the beta/gradient audits, phase-displacement measurements, relaxation-drift
comparisons, and 70 completed bounded EqProp trainings under the revised
collection. Those do not change the coverage of the three proposed tables.

## Sources and verification

All 180 unique bundles used here were read locally: 54 wide trainings,
81 original bounded BPTT trainings, 15 additional longer-T BPTT trainings
and 30 noisy trainings. For each, the successful completion record,
validation-only dataset declaration, manifest and metrics-file hashes,
saved-config/manifest T/K and weight bounds, and ledger best/final values
were checked. The statistics were recomputed from `result.json`, using
sample SD (`n-1`). This metadata/statistics check does not replace the
final scientific reuse seal or repeat the previous checkpoint-level audits.

- [Per-run values, exact settings and source result hashes](../paper_ready_results/three_table_overview_20260916.csv)
- [Current clean ledger](../paper_ready_results/current_contract_run_status.csv)
- [Original shared-T/K ledger](../paper_ready_results/run_status.csv)
- [Longer-T baseline results](../paper_ready_results/baseline_tk_revision/README.md)
- [Completed noise results and comparison limits](../paper_ready_results/read_noise_sweep_results.md)
- [Noise ledger](../paper_ready_results/read_noise_run_status_20260914.csv)
- [Current qualification and scope record](paper_ready_results_manifest.md)
- [Experiment definition and test policy](conv_paper_experiment_definition.md)
- [Current manuscript experimental section](../papers/amplification_overleaf/bidir_paper_theory_revised.tex)
