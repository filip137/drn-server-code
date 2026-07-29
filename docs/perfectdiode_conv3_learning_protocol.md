# Perfect-Diode Conv3 Wide-Range T/K And Rho Protocol

Updated: 2026-07-29

Status: scientific design approved and authorized for agent execution. Agents
may validate the immutable configs/manifests, run the `T/K` and learning-rate
studies, and interpret their results without separate per-stage approval. The
frozen scientific gates, ceilings, routing, and scope below remain binding.

## Scope

This protocol defines a seed-0 ordinary-MNIST diagnostic for:

- Conv3 only;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- perfect diode only;
- plain SGD and Adam;
- operational `T/K` selection followed by a three-epoch parameter-relative-rho
  learning-rate screen.

The `T/K` design namespace is
`mnist-conv-perfectdiode-tk-study/v1`. The downstream learning-rate design
namespace is `mnist-conv-perfectdiode-hparam-study/v2`; v1 Conv1/Conv2 study
and run identities remain immutable.

This ordinary-MNIST study selects `T/K` and the wide-range
parameter-specific LR vector for each matching deterministic medium-affine
paper surface. Its validation accuracy and checkpoints are diagnostics, not
paper-facing evidence. A handoff remains valid only while architecture,
scheme, optimizer, initialization, input gain, `T/K`, and weight contract
match the paper config.

## Frozen ordinary-MNIST contract

- Source data: the 60,000-example MNIST training split.
- Split: the existing deterministic, stratified 55,000/5,000 train/validation
  split, with 500 validation examples per class and split seed `0`.
- Official MNIST test split: never instantiate or read it. Every artifact and
  completion record must contain `official_test_read=false`.
- Affine augmentation and permutation: disabled.
- Preprocessing: normalize as `0.3 * (x - 0.1307) / 0.3081`, then concatenate
  the normalized tensor with its negative to form the signed two-channel
  input.
- Architecture: Conv3 hidden channels `[64,128,256]`, kernels `[3,3,3]`,
  strides `[2,2,1]`, padding `[1,1,1]`, no pooling, and paired 20-output
  squared-error loss.
- Nonlinearity: perfect diode with clamp epsilon `1e-8`. Every source config
  must contain the explicit diode parameter dictionaries required by the
  repository; they must not be supplied by a silent default.
- Conductance-weight contract: wide-range projection to `[0,100]`,
  Kaiming-uniform initialization, and weight gain `1`.
- Input gain: exactly `360`, shared unchanged across all three schemes.
  Record provenance as `user_fixed_ordinary_mnist_diagnostic`,
  `shared_across_schemes=true`, and `recalibrate=false`.
- Model seed and train-loader shuffle seed: independently `0`.
- Initialization: one hash-verified Conv3 checkpoint shared across schemes,
  optimizers, diagnostics, probes, canaries, and candidates. Reset global
  layer and parameter-name counters before each independent build.
- Training batch size: `16`; validation batch size: `64`.
- Equilibrium minimization: exact fixed steps,
  `adaptive_equilibrium=false`, and state reset at the start of every batch.

The split indices, diagnostic cohorts, training minibatch order, shared
initialization, and source/config/environment identities must be materialized
once and referenced by SHA-256 from all downstream artifacts.

## Study 1: select T/K

Select one `T` and one `K` independently for each of the three schemes. The
selected pair is shared between that scheme's SGD and Adam surfaces. A scheme
that does not resolve `T/K` produces explicit zero-work learning-rate outcomes
for both optimizers.

### T from perfect-diode residuals

Use 1,024 deterministic examples from the 55,000-example training subset in
stored loader order, evaluated in batches of `64`.

The core candidate grid is:

```text
T = {4,6,8,10,16,24,32,48,64}
T_ref = 64
```

For every sample and hidden layer, compute the projected KKT residual from
`g=dE/dz` and the settled state `z`. With clamp epsilon `eps=1e-8`:

- for the excitatory/positive half, use `abs(g)` when `z > eps`, otherwise
  `max(-g,0)`;
- for the inhibitory/negative half, use `abs(g)` when `z < -eps`, otherwise
  `max(g,0)`.

The unconstrained output layer uses the raw residual `abs(g)`. Raw hidden
residuals and clamp occupancy are retained as diagnostics but cannot replace
the projected-KKT gate.

For each layer and sample, take the maximum residual over all units; then take
the p90 of those maxima over all 1,024 examples. Select the smallest tested
`T` for which every hidden layer and the output have finite, unrounded
`layer_residual_p90 < 1e-2`. The comparison is strict. `T=64` must also pass
the same gate, even when a smaller candidate passes.

Missing samples, missing layers, non-finite values, a failed `T=64` sentinel,
or no passing core candidate triggers exactly one extension:

```text
T = {96,128,192,256}
```

Preserve the complete core curve. Select the smallest passing value from the
combined ordered grid only if `T=256`, the extension sentinel, also passes.
Otherwise leave the scheme `unresolved_tk`; never assign a fallback `T`.

Record, per layer and candidate, mean, median, p90, p99, maximum, raw residual
statistics, projected-KKT residual statistics where applicable, and clamp
occupancy. Only projected-KKT hidden p90 and raw output p90 select `T`.

### K from gradient convergence

Run this stage only after `T` resolves for the same scheme. Use 256
deterministic examples from the 55,000-example training subset in stored
loader order, evaluated in eight batches of `32`.

```text
K = {4,6,8,16,32,64}
K_ref = 64
gradient_zero_epsilon = 1e-12
```

For every `K`, reuse identical prefetched batches, initial parameters,
selected-`T` free equilibria, targets, state resets, and parameter ordering.
Only the unroll/backpropagation count changes.

For each batch and every `ConvWeight_*`, record gradient L2 norm, gradient
RMS, zero fraction, vector relative error, and cosine against the reference.
The selection statistics are computed from the mean batch gradient norm, mean
batch zero fraction, and mean batch cosine. Define the relative norm delta as:

```text
abs(norm_K - norm_ref) / max(abs(norm_K),abs(norm_ref),1e-30)
```

A candidate passes only when every Conv weight satisfies all three inclusive
gates:

- relative gradient-L2-norm delta `<=0.10`;
- absolute zero-fraction delta `<=0.02`;
- gradient-vector cosine `>=0.90`.

The `K=64` reference itself must be viable for every Conv weight:

- median batch gradient RMS is finite and strictly greater than `1e-12`;
- Q90 batch zero fraction is strictly below `0.99`;
- the nominal pre-projection update unit is finite and strictly positive.

A dead reference leaves the scheme
`unresolved_tk_gradient_viability`; a smaller transient-gradient `K` is not a
substitute. Missing, non-finite, shape-mismatched, or unmatched measurements
fail the candidate.

Select the smallest passing `K`. If that value is the boundary `K=64`, run a
`K=128` sentinel on the same cohort and compare `K=64` against `K=128` using
the same three gates and reference-viability requirements. Freeze `K=64` only
if the sentinel comparison passes; otherwise leave the scheme unresolved.

### T/K operating-point audit and ceiling

Before the LR config is generated, replay the selected `T/K` once through the
same launcher, environment, device type, stored cohorts, and shared
initialization. The replay must reproduce the defining T residual gates, K
gradient gates, and reference viability. A mismatch invalidates the selection
rather than changing `T/K`.

The hard ceiling is 39 T measurements: 27 core measurements plus at most 12
extension measurements. The hard ceiling is 21 K/reference measurements: 18
core/reference measurements plus at most three `K=128` sentinels.

The terminal T/K result has exactly three scheme rows, each selected or
explicitly unresolved, and binds the resolved config, cohort, initialization,
per-candidate measurements, selector output, and completion records by
SHA-256.

## Study 2: optimizer-specific rho screen

Generate the LR config and immutable manifest only after Study 1 is terminal.
Embed each selected `T/K` value and the exact upstream selection and
operating-point-audit SHA-256 values. Freeze and record a second exact
experiment plan before any optimizer probe or canary runs; no separate
approval is required.

Rho selection is independent for all six scheme x optimizer surfaces. Never
transfer target rho values, optimizer units, or raw LRs between schemes or
optimizers.

### Optimizers and rho normalization

- SGD: momentum `0`, weight decay `0`.
- Adam: betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0`,
  `amsgrad=false`, `foreach=false`, `fused=false`, `maximize=false`,
  `capturable=false`, and `differentiable=false`.

Use one optimizer group per scientific parameter and a constant LR vector.
On the same first 32 training minibatches, make nominal-LR-one shadow
proposals while restoring parameters, gradients, and complete optimizer state
before every proposal. Adam shadow proposals use fresh moments, variances,
and step counters; real Adam training retains its state across steps.

For each bounded weight:

```text
u_i = median_batch(
    RMS(delta_W_i_pre_projection_at_nominal_LR_1)
    / RMS(W_i_initial)
)

LR(ConvWeight_i) = rho_conv / u_i
LR(DenseWeight_0) = rho_dense / u_dense
```

Measure each hidden bias's linear Q90 nominal-update unit relative to its
attached weight's initial RMS and use:

```text
LR(Bias_i) = min(
    LR(ConvWeight_i),
    rho_conv / u_bias_i_Q90
)
```

An exactly zero bias unit uses the attached Conv-weight LR. Missing or
non-finite units, or a nonpositive Conv-weight unit, leave the surface
`unresolved_probe`.

Repeat the probe with 64 and then at most 128 minibatches when any parameter's
first-half and second-half unit estimates differ by strictly more than 10%.
A unit still unstable at 128 minibatches leaves the surface unresolved.

### Baseline and ours: requested higher fixed grids

Use this exact 3 x 3 grid independently for baseline-SGD, baseline-Adam,
ours-SGD, and ours-Adam:

```text
rho_conv  = {0.009,0.027,0.081}
rho_dense = {0.03,0.09,0.27}
```

There is no center search or automatic downward substitution for these four
surfaces. Each of the nine declared cells receives its own restarted safety
canary, and a failed cell remains a structured negative result.

### Legacy: safer adaptive core

Legacy uses a different starting grid. For legacy-SGD and legacy-Adam, start
with a restarted canary at:

```text
(rho_conv_center,rho_dense_center) = (0.003,0.01)
```

If the center fails a safety gate, divide both targets by `3` and retry from
the shared initialization. Permit at most six center attempts in total. The
first safe center `(c_conv,c_dense)` defines exactly one 3 x 3 core:

```text
rho_conv  = {c_conv/3,c_conv,3*c_conv}
rho_dense = {c_dense/3,c_dense,3*c_dense}
```

Thus, when the initial center is safe, the exact starting core is:

```text
rho_conv  = {0.001,0.003,0.009}
rho_dense = {0.0033333333333333335,0.01,0.03}
```

Reuse a safe center canary by hash. If no center passes within six attempts,
leave the surface unresolved and publish zero candidate work.

After the core, a plateau confined to a lower edge adds the next
factor-of-three lower value; a plateau confined to an upper edge adds the next
factor-of-three upper value. Expand only implicated axes and include the
corner when both axes expand. Reuse existing cells by hash.

Only one expansion wave is allowed. At most one value is added per axis, so a
surface grows from nine to at most 16 cells. Every new cell receives its own
restarted safety canary. No passing core candidate means no automatic
expansion.

After that one wave, no passing candidate or a passing plateau that remains
confined to an outer edge is `unresolved_boundary`. There is no second
expansion, cross-scheme or cross-optimizer transfer, fallback LR, or long
confirmation.

### Historical basis for the different legacy grid

The reviewed Conv2 core screen found all four baseline/ours surfaces at
`rho_conv=0.009,rho_dense=0.03`, with their passing loss plateaus touching
both upper edges. That supports moving their Conv3 diagnostic one
factor-of-three grid upward; it does not establish that the largest new cell
will be safe or optimal.

Legacy has a different safety record. Conv2 legacy-SGD's
`rho_conv=0.009,rho_dense=0.03` canary failed at step 40 with
`loss_ema_explosion`, while its provisional three-epoch core winner was
`0.003,0.03`. Conv2 legacy-Adam's corresponding canary was safe, but its
provisional winner was `0.009,0.01`. The tracked summary is in the
[Conv2 perfect-diode diagnostic card](../result_registry/cards/diagnostics/perfectdiode-ordinary-mnist-lr-core-screen-20260727-conv2.json);
the failed-canary control is also recorded in the historical Conv2
continuation work retained in Git history.
The immutable legacy-SGD canary result has SHA-256
`73f91fac33558058f4c8e21c25b1c3bcef3b54f3d5b8077b50a63a1bac8d9e6a`.

Because Conv3 is deeper and no comparable perfect-diode Conv3 safety surface
exists, the evidence supports the lower adaptive legacy core above rather
than applying the baseline/ours high grid to legacy.

## Canaries, candidates, and selection

Run a separate restarted 640-step canary for every declared rho cell. Full
candidates restart from the shared initialization and never continue from a
canary. Only safety-clean cells are promoted.

Canaries and candidates fail closed on:

- any non-finite loss, gradient, update, model state, optimizer state, or
  diagnostic;
- after the first 32 steps, loss EMA greater than four times its prior minimum
  for eight consecutive steps;
- bounded-weight gradient RMS greater than 100 times its first-32-step median
  for eight consecutive steps;
- bounded-weight combined-bound occupancy more than `0.20` above its initial
  value for 16 consecutive steps;
- bounded-weight projection efficiency below `0.50` for 16 consecutive steps,
  ignoring numerically zero proposals at epsilon `1e-12`.

Every promoted candidate trains for exactly three epochs:

- 3,438 successful optimizer steps per epoch and 10,314 total;
- constant parameter-specific LR vector;
- no warm-up, decay, scheduler, continuation, or restart;
- complete 5,000-example validation after every epoch.

After every epoch, replay the frozen K cohort and require the high-K reference
gradient-viability gates for every Conv weight. A dense-only classifier is
not a valid Conv LR result.

A candidate passes only if it is safety-admissible, completes exactly 10,314
steps, passes every epochwise Conv-gradient-viability check, and has final
validation accuracy `>=0.90`. Exactly 0.90 passes.

For each surface:

1. Find the minimum final validation loss among passing candidates.
2. Form the inclusive 2% relative-loss plateau.
3. Break ties by higher final validation accuracy, higher median projection
   efficiency, lower maximum rho target, lower target sum, lower `rho_conv`,
   lower `rho_dense`, then deterministic candidate identity.
4. Mark a plateau touching any grid edge `unresolved_boundary`; otherwise
   publish the selected three-epoch diagnostic handoff.

After selection, replay every selected epoch checkpoint on the frozen T and K
cohorts. At both selected `T` and `T=64`, require the defining residual gates.
At selected `K`, require the defining comparison against `K=64`, reference
viability, and the `K=128` comparison when selected `K=64`. Record raw and
projected residuals, clamp occupancy, gradient deltas, vector errors, cosines,
gradient viability, achieved relative and conductance-span updates, bound
occupancy, and projection efficiency.

Any failed post-training T/K gate leaves the surface unresolved. Do not change
`T/K` while retaining the selected LR.

The LR core contains 54 candidate cells, or 162 candidate-epochs: four fixed
nine-cell baseline/ours surfaces plus two nine-cell legacy surfaces. With the
single expansion wave, the absolute hard ceiling is 96 three-epoch candidate
cells, or 288 candidate-epochs. Canary failures reduce full promotions.
Legacy center-search canaries do not authorize extra candidate cells.

## Stages, artifacts, and failure states

The public stage order is:

```text
T/K study:
audit
-> assets
-> tk_core
-> tk_select
-> tk_extension
-> tk_operating_point_audit
-> finalize_tk

LR study:
audit
-> import_tk
-> optimizer_probe
-> rho_canary_core
-> rho_core_candidates
-> select_core
-> rho_canary_expansion
-> rho_expansion_candidates
-> select_expanded
-> post_training_tk
-> finalize_lr
```

Resume only complete candidate outputs. An interrupted Adam candidate without
a complete optimizer-state checkpoint restarts from the shared initialization.

Record the approved scientific config, source commit, environment,
train/validation split, T/K cohorts, minibatch order, shared initialization,
raw T/K measurements, LR vectors, step logs, epoch metrics, safety
diagnostics, and terminal selections in the run directory. Stop on a failed
numerical smoke, failed T/K gate, dataset/initializer mismatch, missing
artifacts, or non-finite values.

Terminal LR reasons include `selected_seed0_ordinary_mnist_three_epoch_screen`,
`unresolved_tk`, `unresolved_tk_gradient_viability`, `unresolved_probe`,
`unresolved_no_safe_center`, `unresolved_no_pass`, `unresolved_boundary`,
`unresolved_dead_gradient`, `unresolved_safety`, and
`unresolved_post_training_tk`. Every one of the six declared surfaces must
have one terminal selected-or-unresolved record.

Publish separate diagnostic comparison cards for T/K and the six-surface LR
screen through the existing result registry. A successful handoff freezes the
LR vector for its matching medium-affine wide-range row, but no
ordinary-MNIST accuracy or checkpoint is a paper result.

## Execution Contract

Prefer `trex` or `jean-zay` for Conv3. A short diagnostic may use `main` or
`akib` only when it fits safely and the same target class, environment, and
scientific contract can be preserved.

Assign every scheme x optimizer surface, including its T/K row, probe,
canaries, candidates, and selector inputs, to one recorded target before
launch. Do not split a surface across hosts. Host choice must not depend on
whether a scheme is baseline, ours, or legacy.

Initialization tensors, dataset split, T/K cohorts, and training-order bytes
must be hash-identical across matched surfaces wherever the protocol requires
shared assets. Run at most one worker per GPU unless a measured packing smoke
establishes a different safe contract.

Use the same runner, config, environment, device type, and output path for the
smoke and production command. Every scheme must pass its exact T/K
operating-point gate before its LR surfaces run. Present the cases, target,
budget, duration, and result path before launching, then proceed without
waiting for per-run approval.

For Jean Zay, pass the repository's fail-closed pre-submit tests and live
canary for the exact staged source, config, wrapper, and resource request.
Copy remote outputs locally and validate hashes before selection.
