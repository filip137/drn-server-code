---
experiment: "exp-023"
evidence: "validated-local"
summary: "All153 once-only official tests validated; new-seed noisy means favor ours over baseline in all12 cells, but over legacy in only3, with Conv3 below legacy throughout."
verdicts: {}
---
# Result: exp023 — Official MNIST test evaluation

## Evidence and coverage

Reviewed October 1, 2026 against the [sealed experiment](../experiments/exp-023-official-test.md).
All **153 selected models** were evaluated and collected: 27 clean controls,
54 noisy seed-0 cases across six etas, and72 noisy seed-1/2 cases across four etas.
Each model used exactly one evaluation of all10000 official MNIST test examples.
No selected model was excluded, retrained or retuned after test access.

The final Conv3 worker completed at **14:39:44 Europe/Paris** on October1.
All three production groups passed queue validation and remote collection.
Total measured worker time, including validation-only capacity probes and failed
attempts, was2601.740seconds (**0.7227 GPU-hours**) against the12GPUh cap.
Accounting: `analysis/queue-accounting.json` below the registered study root.

Result root: `results/read-noise-official-test-20261001-v1/`. Local Conv2 evidence
is in workers/test-local-v1; collected Conv1/Conv3 evidence is in
collected/akib/workers/test-akib-v1 and collected/nom-cool-1/workers/test-nom-cool-1-v2.
Remote evidence is retained. The strict `analysis/analyze.py` audit passed153/153
cases, zero pending groups/cases and zero errors. It verifies canonical artifacts,
sealed source/config/checkpoint identities, unchanged saved parameters, zero optimizer
steps,153 unique active reservations, exact test counts and identical raw test
dataset hashes across all hosts. Total example evaluations:1,530,000, on the same
10000-example test set for each model.

The contract was sealed before any official test access:
`907cf036aafb3bebefe7ef76ed4490e85cc0a7841165f313b8e18275d225da43`.
Checkpoint selection is highest validation accuracy, earliest epoch on ties, over
the complete10/30/30-epoch history. This differs from exp022 fixed-final-epoch
validation reporting. Scientific settings remain centered float64 EqProp training,
seed-0-reference D/F=1/4/6 with frozen betas, and clean nativeT=4/6/8 inference.
Actual seed-1/2 D/F was not remeasured.

All nine validation-only capacity profiles passed batch10000, the full test-set
size. Peak reserved memory was6154/11452/21024MiB for Conv1/Akib, Conv2/local and
Conv3/nom-cool-1, with1024/2300/2300MiB headroom. Repeated capacity endpoints and
128-image validation comparisons against batch64 passed before official testing.

Preserved operational failures: two probes stopped before data access because
staged source lacked Git metadata; real Git snapshots of unchanged source fixed
the provenance step. One Conv3 test attempt reserved access but stopped because
official dataset files were absent. The constructor raised before loading arrays.
Ten hashed evidence files, empty metrics, the loader source and absence of raw/legacy
test data proved zero reads. Its unused reservation was archived, and all immutable
failure artifacts remain. The replacement consumed the first actual test pass.
The strict audit verifies this exception; no partial/completed test was repeated.
Fourteen runner tests, seven collector checks and six reconciliation regressions passed.

## Measurements

Primary noisy test accuracy (%): means over uninterrupted seeds1/2, n=2.
Sample SDs, per-seed values, selected epochs and all153 outcomes are in the
[full tables](../../../../../results/read-noise-official-test-20261001-v1/analysis/report.md).

| Model | Training η | Baseline | Ours | Legacy | Ours − legacy (pp) | Ours − baseline (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Conv1 | 1e-05 | 96.655 | 96.755 | 96.940 | -0.185 | +0.100 |
| Conv1 | 0.0001 | 96.620 | 96.790 | 96.795 | -0.005 | +0.170 |
| Conv1 | 0.0003 | 96.450 | 96.800 | 96.720 | +0.080 | +0.350 |
| Conv1 | 0.0005 | 96.330 | 96.740 | 96.700 | +0.040 | +0.410 |
| Conv2 | 1e-05 | 97.250 | 98.220 | 98.290 | -0.070 | +0.970 |
| Conv2 | 0.0001 | 96.045 | 97.655 | 97.695 | -0.040 | +1.610 |
| Conv2 | 0.0003 | 94.725 | 97.105 | 97.200 | -0.095 | +2.380 |
| Conv2 | 0.0005 | 94.105 | 96.695 | 96.660 | +0.035 | +2.590 |
| Conv3 | 1e-05 | 96.115 | 97.090 | 97.540 | -0.450 | +0.975 |
| Conv3 | 0.0001 | 88.885 | 94.480 | 96.285 | -1.805 | +5.595 |
| Conv3 | 0.0003 | 86.110 | 91.280 | 94.230 | -2.950 | +5.170 |
| Conv3 | 0.0005 | 85.450 | 92.795 | 93.115 | -0.320 | +7.345 |

Clean test accuracy (%), means over seeds0/1/2, n=3:

| Model | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv1 | 96.647 | 96.733 | 96.820 |
| Conv2 | 97.840 | 98.457 | 98.500 |
| Conv3 | 98.383 | 99.010 | 99.153 |

[Noisy test-accuracy JPG](../../../../../results/read-noise-official-test-20261001-v1/analysis/primary_noisy.jpg) ·
[Clean controls JPG](../../../../../results/read-noise-official-test-20261001-v1/analysis/clean_controls.jpg).

The20 historical seed-0 Conv2/3 ours/legacy trajectories that reset Adam/RNG at
epoch10 remain separately labeled. Their best selection spans parent and continuation
history; they do not enter the primary uninterrupted means. Other noisy seed0
cases are supplementary. Seeds1/2 were not trained at eta1e-6 or1e-3, so those
noise levels remain single-seed evidence; no missing replicate is imputed.

## Interpretation

Ours has higher noisy mean test accuracy than baseline in all12 architecture/eta
cells and wins22 of24 individual paired comparisons. The two exceptions are Conv1
seed1 at eta1e-5 and1e-4 (−0.07 and−0.01pp). Baseline gaps grow with noise in the
deeper models; Conv3 mean gaps at eta1e-4/3e-4/5e-4 are+5.595/+5.170/+7.345pp.

Relative to legacy, ours has a higher mean in only3 of12 noisy cells: Conv1 at
3e-4 and5e-4 (+0.080/+0.040pp), and Conv2 at5e-4 (+0.035pp). Across all24
individual pairs, ours wins7, loses16 and ties1. Conv3 mean differences are
−0.450/−1.805/−2.950/−0.320pp in increasing eta order. Thus these held-out
measurements do not establish a general advantage over legacy. Legacy also
exceeds ours in all9 clean paired seeds, although clean mean gaps are small.

These descriptive comparisons support a noise-dependent advantage over baseline
for this frozen comparison, with scheme-specific learning rates and beta settings
retained. They do not isolate amplification from all training choices or establish
statistical significance from two/three seeds. Do not infer a test-optimal noise
level or use these test outcomes to retune the reported surface. Differences from
exp022 reflect both the evaluation split and best-validation versus final checkpoint
selection; they are not a pure validation-to-test shift.

## Decision

Close exp023 with all153 requested official-test outcomes collected, validated and
reviewed. These are official-test results suitable for the paper under the sealed
contract and the stated seed/history labels. No manuscript edit was requested or
performed. The completed study needs no further polling; the shared queue service
remains available for unrelated work. No further test read or training is authorized
by this closeout.
