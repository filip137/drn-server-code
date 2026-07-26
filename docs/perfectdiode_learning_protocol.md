# Perfect-Diode Conv1/Conv2 Ordinary-MNIST Learning Protocol

Updated: 2026-07-26

Status: protocol and implementation frozen and tested; measurements pending.
The gains and Conv1/Conv2 `T/K` values below are user-fixed diagnostic choices,
not calibration measurements.

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

This is an ordinary-MNIST optimization diagnostic. It does not replace the
pending per-scheme deterministic-medium-affine perfect-diode calibration,
select the paper optimizer, or authorize final paper training. The
corresponding hard-sigmoid workflow is defined separately in
[`hardsigmoid_learning_protocol.md`](hardsigmoid_learning_protocol.md).

## Frozen ordinary-MNIST contract

- Dataset: ordinary MNIST without affine augmentation or permutation.
- Source split: the 60,000-example MNIST training split.
- Validation split: deterministic, stratified 55,000/5,000 split with 500
  validation examples per class and dedicated split seed `0`.
- Official MNIST test split: never instantiated or read.
- Preprocessing: the established signed two-channel representation,
  `0.3 * (x - 0.1307) / 0.3081`, concatenated with its negative.
- Architectures: the frozen padding-1 Conv1/Conv2 geometries with paired
  20-output squared-error loss.
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

Every handoff records `official_test_read=false`; no status may imply a frozen
medium-affine or final-paper result.

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

The Conv1 screen runs as an immutable Akib shard and the Conv2 screen as an
immutable Trex shard, with one process per GPU. The shards are copied back and
hash-merged locally; the hosts never mutate the same study directory. All
eligible long confirmations run sequentially on Trex.

Before the screens, the exact committed source archive and config identity
must be verified on both hosts. Each host must pass remote import/plan smoke
tests and one representative Adam center canary while recording memory and
throughput. No screen job is launched from an uncommitted or dirty source
tree.
