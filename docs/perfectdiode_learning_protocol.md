# Perfect-Diode Conv1/Conv2 Wide-Range Rho Protocol

Updated: 2026-07-29

Status: protocol and implementation frozen and tested. Fixed-`T/K` security
checks, probes, canaries, and promoted core candidates are measured. Nine
user-directed interim LR vectors are authorized below for current unbounded
results. Three Conv1/Conv2 surfaces remain without an interim vector, and
selector publication, terminal evidence, and long confirmations remain
incomplete. The gains and Conv1/Conv2 `T/K` values below are user-fixed
diagnostic choices, not calibration measurements.

## Scope

This protocol defines the ordinary-MNIST seed-0 hyperparameter workflow for:

- Conv1 and Conv2;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- perfect diode only;
- plain SGD and Adam.

The design namespace is `mnist-conv-perfectdiode-hparam-study/v1`; candidate
runs use `mnist-conv-perfectdiode-hparam-run/v1`. The canonical config is
[`perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json`](../configs/conv/perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json),
with content-derived study identity
`lrstudy_5afe8bc7c9180cd1c677d1d87a497e1d6a89c1599cd015d7a6c127ebb18f2e46`.
Existing LR study and run identities v1-v7 remain immutable.

This ordinary-MNIST study selects the wide-range parameter-specific LR vector
for each matching deterministic medium-affine paper surface. Its validation
accuracy is an optimization diagnostic, not paper-facing evidence. The
handoff is valid only when architecture, scheme, optimizer, initialization,
input gain, `T/K`, and weight contract match the paper config exactly.

## Frozen ordinary-MNIST contract

- Dataset: ordinary MNIST without affine augmentation or permutation.
- Source split: the 60,000-example MNIST training split.
- Validation split: deterministic, stratified 55,000/5,000 split with 500
  validation examples per class and dedicated split seed `0`.
- Official MNIST test split: never instantiated or read.
- Preprocessing: the established signed two-channel representation,
  `0.3 * (x - 0.1307) / 0.3081`, concatenated with its negative.
- Architectures: Conv1 channels `[64]`, strides `[2]`; Conv2 channels
  `[64,128]`, strides `[2,2]`; kernel `3`, padding `1`, no pooling, and paired
  20-output squared-error loss.
- Conductance-weight contract: wide-range projection to `[0,100]` with the
  reference Kaiming-uniform initialization.
- Model seed and train-loader shuffle seed: independently fixed to `0`.
- Global layer and parameter name counters: reset before every independent
  model build.
- Initialization: one hash-verified checkpoint per architecture, shared
  across schemes, optimizers, probes, and candidates wherever tensor geometry
  agrees.
- Training batch size: `16`; validation batch size: `64`.
- Minimization: exact fixed steps with `adaptive_equilibrium=false` and state
  reset at the start of every batch.
- Perfect-diode clamp epsilon: `1e-8`.

Every artifact records the split indices, minibatch order, initialization
checkpoint, resolved config, source tree, and parent-artifact hashes.

## Fixed input gains and operating points

The following choices apply identically to baseline, ours, and legacy:

| Architecture | Fixed `input_gain` | Operational `T/K` | Provenance |
|---|---:|---:|---|
| Conv1 | `40` | `4/4` | `user_fixed_ordinary_mnist_diagnostic` |
| Conv2 | `100` | `6/6` | `user_fixed_ordinary_mnist_diagnostic` |

These choices have `shared_across_schemes=true` and `recalibrate=false`. They
must not be described as measurements from the paper's 30% clamped-occupancy
calibration.

Before any rho probe, each fixed row undergoes only the fixed-T/K gradient
security check below. Conv3 perfect-diode selection is outside this v1
pipeline and remains unresolved.

## Conv1/Conv2 fixed-T/K gradient security check

The Conv1 and Conv2 `T/K` values are already fixed; they are not reselected.
Before any optimizer probe, perform one scheme-specific check that the fixed
operating point gives approximately the same Conv gradients as the
`T=64,K=64` reference:

- Conv1: compare `(T=4,K=4)` with `(T=64,K=64)`;
- Conv2: compare `(T=6,K=6)` with `(T=64,K=64)`.

Use the same 256-example cohort, batches, initialization, state-reset policy,
targets, and parameter ordering for both sides. Each side computes its own
free equilibrium at its declared `T`. Every `ConvWeight_*` must satisfy:

- relative gradient-L2-norm delta `<=10%`;
- absolute zero-fraction delta `<=2%`; and
- gradient-vector cosine `>=0.90`.

This is only a security check on the fixed `T/K`. It does not rerun selection,
impose a clamped-occupancy gate, or add epochwise T/K audits. Failure produces
`unresolved_fixed_tk_gradient_mismatch` without changing the fixed `T/K`.

## Optimizers and optimizer-specific rho units

Rho selection is independent for every architecture x scheme x optimizer
surface. Targets and raw LRs are never transferred across schemes or
optimizers.

- SGD: momentum `0`, weight decay `0`.
- Adam: betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0`,
  `amsgrad=false`, `foreach=false`, `fused=false`, `maximize=false`,
  `capturable=false`, and `differentiable=false`.

Use one optimizer parameter group per scientific parameter and a constant LR
vector. Make nominal-LR-one shadow proposals on the same first 32 training
minibatches. Restore parameters and complete optimizer state before every
proposal. Adam shadow proposals start with fresh moments, variances, and step
counter; real Adam training retains its state across steps.

For each bounded weight:

```text
u_i = median_batch(
    RMS(delta_W_i_pre_projection_at_nominal_LR_1)
    / RMS(W_i_initial)
)

LR(ConvWeight_i) = rho_conv / u_i
LR(DenseWeight_0) = rho_dense / u_dense
```

This gives every Conv weight and the Dense weight an independent raw LR while
searching only `rho_conv` and `rho_dense`.

Because perfect-diode biases directly affect clamping, measure a linear Q90
bias-update unit normalized by its attached weight's initial RMS and use:

```text
LR(Bias_i) = min(
    LR(ConvWeight_i),
    rho_conv / u_bias_i_Q90
)
```

If the bias unit is exactly zero, use the attached Conv-weight LR. Missing or
non-finite units leave the row unresolved.

Repeat the probe with 64 and then at most 128 minibatches if any parameter's
first-half and second-half unit estimates differ by more than 10%. A unit
still unstable at 128 minibatches, or any nonpositive Conv-weight unit, leaves
the row unresolved.

## Rho core and restarted canaries

Start every row x optimizer surface at:

```text
(rho_conv_center, rho_dense_center) = (3e-3, 1e-2).
```

Run a restarted 640-step constant-LR canary. If the center fails a safety
gate, divide both targets by `3` and retry from the shared initialization.
Permit at most six downward center attempts.

For the first safe center `(c_conv,c_dense)`, form:

```text
rho_conv  = {c_conv/3, c_conv, 3*c_conv}
rho_dense = {c_dense/3, c_dense, 3*c_dense}
```

Run a separate restarted 640-step canary for every core cell. Only
safety-clean cells are promoted. Full candidates restart from initialization
and never continue from canaries. The 640-step gate covers both early failures
around step 40 and later failures around step 586 seen previously.

## Three-epoch candidates and safety

Every promoted candidate trains for exactly three epochs:

- 3,438 optimizer steps per epoch and 10,314 successful steps total;
- constant parameter-specific LR vector;
- no warm-up, decay, scheduler, or restart;
- full 5,000-example validation after every epoch.

The three-epoch requirement applies to every LR cell that passes its 640-step
canary and is promoted. A cell that fails the canary stops there and does not
consume three candidate epochs.

Canaries and candidates use these fail-closed gates:

- any non-finite loss, gradient, update, state, or diagnostic;
- after the first 32 steps, loss EMA greater than four times its prior minimum
  for eight consecutive steps;
- bounded-weight gradient RMS greater than 100 times its first-32-step median
  for eight steps;
- bounded-weight combined-bound occupancy more than `0.20` above its initial
  value for 16 steps;
- bounded-weight projection efficiency below `0.50` for 16 steps, ignoring
  numerically zero proposals.

A candidate passes only when it is safety-admissible, completes exactly
10,314 steps, and reaches final validation accuracy `>=90%`. Exactly 90%
passes.

## Selection and one symmetric expansion

For each row x optimizer surface:

1. find the minimum final validation loss among passing candidates;
2. form the inclusive 2% loss plateau;
3. break ties by higher accuracy, higher projection efficiency, lower maximum
   target, lower target sum, lower `rho_conv`, lower `rho_dense`, then
   deterministic candidate identity.

A plateau confined to a lower edge adds the next factor-of-three lower value;
a plateau confined to an upper edge adds the next factor-of-three upper
value. Expand only implicated axes and include the corner when both expand.
Reuse existing cells by hash.

Only one expansion wave is allowed. At most one value is added per axis, so a
surface grows from 9 to at most 16 cells. Each new cell must pass its own
restarted canary. No passing core candidate means no automatic expansion.

After that one wave, no passing candidate or a plateau still confined to an
outer edge leaves the surface unresolved. There is no second expansion,
cross-scheme transfer, optimizer transfer, or fallback LR.

## Executed core screen and conclusion

The seed-0 core execution used source commit
`093d3bc325a5ecb92262639798408a845d9767f1` and the immutable study identity
above. All six fixed-`T/K` security rows passed their comparison with
`(T=64,K=64)`. Both hosts completed assets, optimizer probes, all core
canaries, and 53 promoted candidate entries each. Across the two
architectures, 104 candidates were safety-admissible and 101 reached the
inclusive 90% accuracy gate.

The selector then stopped during `select_core` because a structured
projection-efficiency failure was passed to an interface that required a
string. Before stopping, it published only the Conv1 baseline-SGD and Conv2
baseline-SGD `needs_expansion` entries. Candidate measurements remain intact.
The table below applies the intended selector read-only to all twelve
surfaces: those two entries reproduce their published values, while the other
ten are reconstructions. Every rho pair remains a diagnostic core winner
rather than a frozen handoff.

| Architecture | Scheme | Optimizer | Passing / trained | Diagnostic core rho `(conv,dense)` | Final validation loss / accuracy | Core decision |
|---|---|---|---:|---:|---:|---|
| Conv1 | baseline | SGD | `9/9` | `(0.009,0.03)` | `0.10242748 / 95.44%` | expand Dense upward |
| Conv1 | baseline | Adam | `8/9` | `(0.009,0.03)` | `0.09903789 / 95.96%` | expand Dense upward |
| Conv1 | proposed/ours | SGD | `9/9` | `(0.009,0.03)` | `0.09703515 / 95.94%` | expand Dense upward |
| Conv1 | proposed/ours | Adam | `9/9` | `(0.009,0.03)` | `0.09543736 / 96.02%` | expand Conv upward |
| Conv1 | legacy | SGD | `5/8` | `(0.003,0.03)` | `0.12925158 / 94.46%` | expand Dense upward |
| Conv1 | legacy | Adam | `9/9` | `(0.003,0.01)` | `0.09618260 / 96.18%` | core bracketed |
| Conv2 | baseline | SGD | `9/9` | `(0.009,0.03)` | `0.08552010 / 95.70%` | expand both upward |
| Conv2 | baseline | Adam | `8/9` | `(0.009,0.03)` | `0.08505717 / 95.90%` | expand both upward |
| Conv2 | proposed/ours | SGD | `9/9` | `(0.009,0.03)` | `0.06503381 / 97.18%` | expand both upward |
| Conv2 | proposed/ours | Adam | `9/9` | `(0.009,0.03)` | `0.06219476 / 97.18%` | expand both upward |
| Conv2 | legacy | SGD | `8/8` | `(0.003,0.03)` | `0.07388557 / 96.78%` | expand both upward |
| Conv2 | legacy | Adam | `9/9` | `(0.009,0.01)` | `0.05470467 / 97.78%` | expand Conv upward |

The core accuracies are generally strong: every diagnostic winner is above
94%, and the requested five-row Conv2 continuation lies between 95.70% and
97.18% after only three epochs. Eleven of the twelve surfaces nevertheless
have a passing 2% loss plateau confined to an upper search edge. The useful
rho region is therefore not bracketed. The evidence supports testing larger
rho targets, but does not yet establish that the largest safe rho is optimal.

The requested next diagnostic scope is deliberately limited to Conv2
baseline and ours under both optimizers plus legacy SGD. Every one of those
five surfaces adds `rho_conv=0.027` and `rho_dense=0.09`, producing seven new
Cartesian cells per surface while reusing the nine core cells by hash. Conv2
legacy Adam and all Conv1 extensions are outside this requested next batch.
No extension, final selection, or long-confirmation run has been launched.

The full-precision reconstructed table is
[`conv_perfectdiode_lr_core_screen_20260727.csv`](conv_perfectdiode_lr_core_screen_20260727.csv).
Its SHA-256 is
`cd8228bcc1096b733ef7c729622fc5e9f53911489767822a9ca6fb84012da870`.
These measurements remain ordinary-MNIST optimization diagnostics. They do
not resolve deterministic-medium-affine perfect-diode gain calibration,
operational `T/K`, optimizer choice, or final paper performance. The incomplete
core screen also does not yet freeze a terminal learning-rate handoff; a vector
becomes authoritative only after the selection and confirmation stages below
resolve its full parameter-specific surface.

## Interim unbounded-result LR vectors

Filip supplied the following interim vectors on 2026-07-29. For now, use each
vector unchanged for its matching wide-range/unbounded result row. This is an
explicit user-directed interim handoff; it does not retrospectively make the
incomplete selector terminal.

Notation maps directly to the scientific parameter names:

```text
C_i = ConvWeight_i
B_i = Bias_i
D   = DenseWeight_0
```

| Architecture | Scheme | Optimizer | Parameter-wise LR vector |
|---|---|---|---|
| Conv1 | baseline | SGD | `C0=B0=0.142696, D=0.0120238` |
| Conv1 | baseline | Adam | `C0=B0=8.66763e-4, D=1.09401e-4` |
| Conv1 | ours | SGD | `C0=B0=0.0375994, D=0.00311655` |
| Conv1 | ours | Adam | `C0=B0=8.66753e-4, D=1.09388e-4` |
| Conv1 | legacy | SGD | `C0=B0=2.05753e-4, D=4.76259e-5` |
| Conv2 | baseline | SGD | `C0=B0=7.90864, C1=B1=3.81476, D=0.820630` |
| Conv2 | baseline | Adam | `C0=B0=2.60078e-3, C1=B1=4.59053e-4, D=4.67660e-4` |
| Conv2 | legacy | SGD | `C0=B0=0.00496733, C1=0.00367839, B1=6.74726e-4, D=0.00236044` |
| Conv2 | legacy | Adam | `C0=B0=8.66750e-4, C1=B1=1.52841e-4, D=5.16982e-5` |

No interim vector was supplied for Conv1 legacy Adam, Conv2 ours SGD, or
Conv2 ours Adam. Leave those rows unresolved; do not recover values for them
from an older config, diagnostic winner, or unpublished selector output.

## Stages, resume, and output

The public stage order is:

```text
audit
-> assets
-> fixed_tk_gradient_security
-> optimizer_probe
-> rho_canary_core
-> rho_core_candidates
-> select_core
-> rho_canary_extension
-> rho_extension_candidates
-> select_final
-> long_confirm
-> finalize
```

Conditional extension and confirmation stages publish explicit zero-work
completions. Manifests are immutable and embed upstream hashes. Resume skips
only hash-verified complete entries. An interrupted Adam run without a
complete optimizer-state checkpoint restarts from initialization.

SGD and Adam produce separate handoffs containing the complete operating
point, target pair, ordered raw LR vector, epochwise metrics, safety
diagnostics, and all provenance hashes. Successful rows use
`selected_seed0_ordinary_mnist_three_epoch_screen`. Unresolved reasons include
`unresolved_no_pass`, `unresolved_boundary`, `unresolved_probe`,
and `unresolved_fixed_tk_gradient_mismatch`.

Every handoff records `official_test_read=false`. A successful handoff freezes
the LR vector for its matching medium-affine wide-range row, but no
ordinary-MNIST accuracy or checkpoint is a paper result.

## Long confirmations

Every resolved scheme x optimizer surface receives one fresh confirmation
from the shared seed-0 initialization. Conv1 runs for exactly 10 epochs
(`34,380` optimizer steps); Conv2 runs for exactly 30 epochs (`103,140`
optimizer steps). The selected LR vector stays constant, validation covers
all 5,000 held-out training examples after every epoch, and both the
best-validation-loss checkpoint and final checkpoint are retained.

A surface receives an explicit zero-work long-confirm completion when it has
no candidate at or above 90%, remains boundary-unresolved after its one
expansion, or fails security or provenance checks. Confirmations are never
continued from a three-epoch candidate.

## Cost ceiling and execution

This protocol has 12 rho surfaces:
`2 architectures x 3 schemes x 2 optimizers`. The nine-cell core permits at
most 108 full candidates, or 324 candidate-epochs. If every surface expands
to 16 cells, the absolute ceiling is 192 candidates, or 576
candidate-epochs. Canary failures reduce full promotions.

Prefer Conv1 surfaces on `main` or `akib` and Conv2 surfaces on `akib` or
`trex`, with one process per GPU. Assign each scientific surface to one
recorded target before launch. Copy remote shards back and hash-merge locally;
hosts never mutate the same study directory. Prefer Trex for eligible long
confirmations.

Before a screen, verify the exact committed source archive and config identity
on its target. Every used target must pass import/plan smoke tests and one
representative Adam center canary while recording memory and throughput. No
screen job is launched from an uncommitted or dirty source tree.
