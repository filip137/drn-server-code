# IBM OM fixed-identity local reference compensation

## Scientific question

Can a four-device weight obtain a true logical zero without nonlocal device
reassignment by programming each existing cell to a nearby stored baseline?

This study separates two quantities that must not be conflated:

- `r_i` is the immutable intrinsic symmetry-point parameter of device identity
  `i` in the fitted IBM OM model.
- `B_i` is a conductance target that the initialization procedure proposes to
  store on that same physical identity.

The treatment changes `B_i`; it does not change `r_i`, exchange identities, or
move any other fitted device field.

This is the fixed-binding local member of the
[symmetry-reference-anchored study family](ibm_om_reference_anchored_studies.md).

## Frozen initialization inputs

Both native arms receive the same explicit pre-BPTT logical checkpoint at
`data/ibm_om_cell_aware_full_span_v1.pt` through `--weights` (SHA-256
`a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563`)
and the same teacher at `data/mnist_relu_teacher_fixed_init_20260816.pt`
through `--teacher-weights` (SHA-256
`9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`).
The checkpoint derivation is frozen by
`data/ibm_om_cell_aware_full_span_v1.receipt.json` (SHA-256
`d58b0d709312061dc2425d143744f45fdf9de9d5e583a619e7a91325c53059b8`).
No newest-checkpoint discovery or substitution is permitted.

## Fixed identity binding

Each assignment samples one device identity at every physical address.  The
sampled flat order and the existing groups of four are frozen for both arms.
The four positions have sign order

\[
s=(+1,-1,-1,+1)
\quad\text{for}\quad (++,+-,-+,--).
\]

Unlike the preceding reference-balanced-binding screen, this study performs
no sorting, quartet regrouping, donor selection, permutation, or reassignment.
Identity and binding hashes must be identical between policies.

## Baseline policies

Map the fitted symmetry point to the normalized circuit coordinate,

\[
\rho_i=\operatorname{clip}((r_i+1)/2,0,1),
\]

and let `[l_i,u_i]` be that identity's sampled active interval.

The control is the nearest individually attainable symmetry baseline,

\[
B_i^{(0)}=\operatorname{clip}(\rho_i,[l_i,u_i]).
\]

The treatment is the unique minimum-displacement box projection with exact
zero signed contrast in each existing quad,

\[
B^*=\arg\min_B \frac12\lVert B-\rho\rVert_2^2
\quad\text{subject to}\quad
l_i\le B_i\le u_i,\qquad s^\mathsf{T}B=0.
\]

Its KKT form is

\[
B_i(\lambda)=\operatorname{clip}(\rho_i-\lambda s_i,[l_i,u_i]).
\]

A deterministic monotone bisection solves the single scalar equation
`s^T B(lambda)=0`.  Without an active bound, the solution reduces to the
checkerboard correction

\[
B^*=\rho-\frac{s^\mathsf{T}\rho}{4}s.
\]

That unconstrained correction preserves both row sums, both column sums, and
total zero-state loading.  When a bound is active, the declared objective is
still minimum L2 displacement plus exact zero; any resulting row, column, or
total loading change is measured rather than silently constrained away.

Here, exact zero means the normalized float64 baseline target satisfies the
constraint to the declared tolerance.  Conversion to the canonical float32
physical conductances can leave a last-bit residual; that physical residual is
measured and retained in the mapping report and is never subtracted from the
circuit equations.

Exact zero is feasible for a quad precisely when zero lies between its minimum
and maximum attainable signed sums:

\[
\sum_{s_i=+1}l_i-\sum_{s_i=-1}u_i\le0\le
\sum_{s_i=+1}u_i-\sum_{s_i=-1}l_i.
\]

A read-only preflight over the development assignment and three held-out
assignments found all 158,800 quads feasible.  The unconstrained checkerboard
solution already stayed within bounds for 97.66% of quads; 2.34% required the
box-constrained fallback.  These facts establish continuous feasibility only.
They do not establish that the required baseline trim is distinguishable under
a pulse codebook or program-and-verify controller.

## Matched continuous weight mapping

Every physical branch is represented explicitly as

\[
G_i=B_i+d_i,\qquad d_i\ge0.
\]

No baseline is subtracted before circuit evaluation.  For each quad, the full
conductances determine both signed transfer and loading,

\[
C_Q=\frac{G_{++}-G_{+-}-G_{-+}+G_{--}}{2},
\qquad
L_Q=G_{++}+G_{+-}+G_{-+}+G_{--}.
\]

Thus the denominator retains the sum of all physical conductances, including
all four `B_i` terms.  Positive logical weights raise the diagonal pair and
negative logical weights raise the off-diagonal pair; inactive offsets remain
zero.

To isolate the baseline intervention, define one common headroom per existing
quad as the minimum positive headroom across all four cells and both baseline
policies.  At a given layer-scale pair, construct the logical offset once from
that common headroom and use the identical `d` tensor in the control and
treatment.  Both target hashes and canonical physical-offset hashes must
match.  A policy is not allowed to gain accuracy through extra headroom.

This first study is continuous and ideal.  Quantized levels, a four-delta
spacing, pulse limits, P&V, write noise, read noise, drift, HWA, QAT/BPTT, and
on-chip updates are absent by construction.

## Calibration and comparisons

Assignment 86001 is the only development population.  Each baseline policy
searches the same 16 layer-scale pairs from
`{0.125,0.25,0.5,1.0}^2`, selecting by calibration accuracy, calibrated KL,
then lexicographic pair, and fitting the same positive logit-gain grid.  Both
calibrations are frozen before assignments 87001, 87002, and 87003 are
evaluated.

The held-out evaluation retains the complete two-by-two
baseline-policy-by-calibration-policy matrix.  The primary causal comparison
uses the control calibration for both physical policies:

`local_min_l2_exact_zero@cal_local_nearest_symmetry`
minus
`local_nearest_symmetry@cal_local_nearest_symmetry`.

This prevents refitting from being mistaken for the baseline effect.  The
secondary attainable comparison is the scheme-refitted treatment minus the
scheme-refitted control.  Both cross-calibration cells remain reported.

The mechanistic gate requires a held-out primary mean gain of at least five
percentage points.  The absolute recovery gate requires at least 90% held-out
mean accuracy for the scheme-refitted treatment.  A failed gate is preserved
as evidence; it is not repaired after inspecting test results.

## Required artifacts and checks

Each native arm must record enough information to audit the intervention:

- source checkpoint and teacher hashes, AIHWKit version, assignment seeds,
  population fingerprints, and fixed address-to-identity hashes;
- intrinsic `r`, mapped `rho`, active bounds, control `B`, treatment `B`, the
  correction, KKT multiplier, bound-active mask, and feasibility interval;
- exact-zero residuals, lower/upper-bound residuals and hit counts, correction
  distributions, and row, column, total, and signed loading changes;
- common headroom, zero-capacity counts, logical offset targets, canonical
  physical `B`, `d`, and `G`, their hashes, and maximum `G-(B+d)` residual;
- scale-search results, selected calibrations, full held-out prediction hashes,
  exact correct counts, accuracy, KL, voltage RMS, and sign diagnostics; and
- a main/replay comparison that permits only run-local paths and run IDs to
  differ.

The metric identifier is
`ibm_om.local_reference_compensated_continuous_init.v1`.  Its evidence class is
`model_based_aihwkit_preset`: it is a fitted-model ideal-initialization result,
not measured fabricated-array accuracy and not a claim about a deployable P&V
sequence.

## Artifact-verified results

Both canonical native arms completed with 53 artifacts each.  Full artifact
hash verification and independent main/replay reconstruction passed.  On the
three held-out assignments, the scheme-refitted ideal continuous results were:

| Assignment | Nearest symmetry | Local compensation | Paired gain |
|---|---:|---:|---:|
| 87001 | 84.23% | 90.73% | +6.50 pp |
| 87002 | 87.30% | 93.58% | +6.28 pp |
| 87003 | 79.20% | 91.10% | +11.90 pp |
| **Mean** | **83.58%** | **91.80%** | **+8.23 pp** |

The primary comparison held the nearest-symmetry calibration fixed for both
baseline policies.  Its mean improvement was 8.23 percentage points, above
the predeclared 5-point mechanistic gate.  The independently refitted local
compensation mean was 91.80%, above the predeclared 90% absolute gate.  The
same canonical physical `d` hashes were used in every paired comparison, so
the accuracy difference is attributable to the baseline policy within this
ideal continuous contract rather than extra treatment headroom.

These measurements do not establish that the fine baseline correction can be
resolved by discrete pulses, survives stochastic P&V, or improves on-chip
learning.  Those are separate progression gates.

## Native workflow commands

After freezing a clean source commit, prepare the declared study and run both
native arms serially from the same environment:

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1.json \
  --results-root results

EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python3.12 \
python -m ebl validate \
  --config examples/mnist_relu_drn/ibm_om_local_reference_compensation/continuous.json \
  --output-dir results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1/runs/main \
  --weights data/ibm_om_cell_aware_full_span_v1.pt \
  --teacher-weights data/mnist_relu_teacher_fixed_init_20260816.pt

EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python3.12 \
python -m ebl validate \
  --config examples/mnist_relu_drn/ibm_om_local_reference_compensation/continuous.json \
  --output-dir results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1/runs/replay \
  --weights data/ibm_om_cell_aware_full_span_v1.pt \
  --teacher-weights data/mnist_relu_teacher_fixed_init_20260816.pt
```

Then reconstruct the evidence independently and summarize the workflow:

```bash
python -m experiments.mnist_relu_drn.analyze_ibm_om_local_reference_compensation \
  --main-run results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1/runs/main \
  --replay-run results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1/runs/replay \
  --output-dir results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1/analysis/local_reference_compensation

python -m ebl study summarize \
  --study-dir results/mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1 \
  --verify-artifacts
```

## Decision boundary

If exact local zero improves accuracy, the next matched test applies the same
fixed identities and compensated baseline to the standard four-delta codebook,
followed by deterministic pulse reachability and then stochastic P&V.  The
continuous result alone cannot show that small baseline corrections can be
written.

If exact local zero does not recover accuracy, the full-conductance diagnostics
must be used to distinguish loading, headroom, bound-active corrections, and
calibration effects before changing level spacing or invoking training.
