# Hard-Sigmoid Conv1/2/3 Ordinary-MNIST Learning Protocol

Updated: 2026-07-26

Status: protocol frozen for implementation; concrete gain imports, config
identity, and measurements pending.

## Scope

This protocol defines the ordinary-MNIST seed-0 hyperparameter workflow for:

- Conv1, Conv2, and Conv3;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- hard sigmoid only;
- plain SGD and Adam.

The design namespace is `mnist-conv-hardsigmoid-hparam-study/v1`; candidate
runs use `mnist-conv-hardsigmoid-hparam-run/v1`. A concrete config and its
content-derived identity must exist before execution. Existing LR study and
run identities v1-v7 remain immutable.

This is an ordinary-MNIST optimization diagnostic. It does not replace the
deterministic-medium-affine paper handoff, select the paper optimizer, or
authorize final paper training. The corresponding perfect-diode workflow is
defined separately in
[`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md).

## Frozen ordinary-MNIST contract

- Dataset: ordinary MNIST without affine augmentation or permutation.
- Source split: the 60,000-example MNIST training split.
- Validation split: deterministic, stratified 55,000/5,000 split with 500
  validation examples per class and dedicated split seed `0`.
- Official MNIST test split: never instantiated or read.
- Preprocessing: the established signed two-channel representation,
  `0.3 * (x - 0.1307) / 0.3081`, concatenated with its negative.
- Architectures: the frozen padding-1 Conv1/Conv2/Conv3 geometries with paired
  20-output squared-error loss.
- Hard sigmoid: `g_on=100`, `g_off=0`, `v_off=4.0`, active interval `[-4,4]`.
- Model seed and train-loader shuffle seed: independently fixed to `0`.
- Global layer and parameter name counters: reset before every independent
  model build.
- Initialization: one hash-verified checkpoint per architecture, shared
  across schemes, optimizers, probes, and candidates wherever tensor geometry
  agrees.
- Training batch size: `16`; validation batch size: `64`.
- Minimization: exact fixed steps with `adaptive_equilibrium=false` and state
  reset at the start of every batch.

Every artifact records the split indices, minibatch order, initialization
checkpoint, resolved config, source tree, and parent-artifact hashes.

## Input-gain prerequisite

Input gain is fixed before this protocol and is never selected from
LR-candidate accuracy. The concrete config must contain one explicit gain for
each architecture x scheme row.

The initial implementation may import the nine currently frozen
deterministic-medium-affine gains only when it records their source hashes and
states that they are intentionally reused unchanged for this ordinary-MNIST
diagnostic. Silent reuse or relabeling as ordinary-MNIST calibration is
forbidden.

## T selection

Select `T` independently for every architecture x scheme row. The selected
value is shared between SGD and Adam.

- Cohort: 1,024 deterministic examples from the 55,000-example training
  subset.
- Candidate grid: `T={4,6,8,10,16,24,32,48,64}`.
- Mandatory reference/sentinel: `T_ref=64`.
- Optional one-time extension: `T={96,128,192,256}`.

For every layer and sample, take the maximum raw equilibrium residual
`abs(dE/dz)` over units, then the 90th percentile over samples. A candidate
passes only when every hidden and output layer has a finite
`layer_residual_p90 < 1e-2`. The comparison is strict and unrounded.

Select the smallest passing `T` only if `T=64` also passes. If no core value
passes, or the sentinel fails, evaluate the extension once. A row still
unresolved after `T=256` remains unresolved; no fallback is assigned.

Measure and report the fraction of every hidden layer outside `[-4,4]` at
`T=64` and at the selected `T`. Do not change the fixed input gain from this
measurement.

## K selection and gradient viability

Run this stage after selecting `T` for the same row.

- Cohort: 256 deterministic examples from the 55,000-example training subset.
- Evaluation batch size: `32`.
- Candidate grid: `K={4,6,8,16,32,64}`.
- Reference: `K_ref=64`.
- Conditional sentinel: `K=128` when the smallest passing value is `K=64`.

Use identical prefetched batches, initial parameters, free equilibria,
targets, and parameter ordering for every `K`. Every `ConvWeight_*` must meet:

- relative gradient-L2-norm delta from the reference `<=10%`;
- absolute zero-fraction delta from the reference `<=2%`; and
- gradient-vector cosine with the reference `>=0.90`.

Select the smallest passing `K`. If `K=64` is selected, freeze it only when it
also passes against `K=128`.

The reference itself must be viable for every Conv weight:

- median batch gradient RMS finite and greater than `1e-12`;
- Q90 batch zero fraction strictly below `0.99`; and
- nominal pre-projection update unit finite and positive.

A dead reference leaves the row `unresolved_tk_gradient_viability`. A smaller
transient-gradient `K` is not a substitute for a dead high-`K` reference.
Repeat gradient viability after every training epoch; a dense-only classifier
cannot supply a Conv LR handoff.

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
searching only the two targets `rho_conv` and `rho_dense`.

For a hidden bias, measure a linear Q90 nominal-update unit normalized by its
attached weight's initial RMS and use:

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
safety-clean cells are promoted. Full candidates restart from the shared
initialization and never continue from a canary. The 640-step gate covers both
the early failures around step 40 and later failures around step 586 observed
in earlier diagnostics.

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
10,314 steps, passes epochwise Conv-gradient viability, and reaches final
validation accuracy `>=90%`. Exactly 90% passes.

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

## Post-training T/K audit

Replay every selected epoch checkpoint on the frozen T/K cohorts. Report raw
layer residuals, hard-sigmoid saturation, gradient deltas and cosines,
gradient viability, achieved relative and conductance-span updates, boundary
occupancy, and projection efficiency.

If the selected `T/K` no longer passes its defining gates, leave the handoff
unresolved. Do not silently change `T/K` while retaining the selected LR.

## Stages, resume, and output

The public stage order is:

```text
audit
-> assets
-> tk_core
-> tk_select
-> tk_extension
-> tk_operating_point_audit
-> optimizer_probe
-> rho_canary_core
-> rho_core_candidates
-> select_core
-> rho_canary_extension
-> rho_extension_candidates
-> post_training_tk
-> finalize
```

Conditional stages publish explicit zero-work completions. Manifests are
immutable and embed upstream hashes. Resume skips only hash-verified complete
entries. An interrupted Adam run without a complete optimizer-state
checkpoint restarts from initialization.

SGD and Adam produce separate handoffs containing the complete operating
point, target pair, ordered raw LR vector, epochwise metrics and audits, and
all provenance hashes. Successful rows use
`selected_seed0_ordinary_mnist_three_epoch_screen`. Unresolved reasons include
`unresolved_no_pass`, `unresolved_boundary`, `unresolved_probe`,
`unresolved_tk`, and `unresolved_tk_gradient_viability`.

Every handoff records `official_test_read=false`; no status may imply a frozen
medium-affine or final-paper result.

## Cost ceiling and execution

This protocol has 18 possible rho surfaces:
`3 architectures x 3 schemes x 2 optimizers`. The nine-cell core permits at
most 162 full candidates, or 486 candidate-epochs. If every surface expands
to 16 cells, the absolute ceiling is 288 candidates, or 864
candidate-epochs. Canary failures reduce full promotions.

Execution may proceed in independently finalized architecture blocks. Before
any Jean Zay batch or array is submitted, the fail-closed Jean Zay pre-submit
test and live-canary gate must pass for the exact staged source, config,
environment, wrapper, and resource request.
