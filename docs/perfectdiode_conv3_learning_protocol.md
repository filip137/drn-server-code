# Perfect-Diode Conv3 Ordinary-MNIST T/K and Learning-Rate Protocol

Updated: 2026-07-27

Status: scientific design approved for implementation. No experiment has been
authorized or launched by this document. Execution remains blocked on
validated immutable configs/manifests and separately approved exact experiment
plans for the `T/K` and learning-rate studies.

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

This is an ordinary-MNIST optimization diagnostic. It does not calibrate a
perfect-diode paper gain, fill the deterministic-medium-affine paper grid,
select the paper optimizer, authorize long confirmation, or authorize final
paper training.

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
- Conductance bounds and initialization: `[0,100]`, Kaiming-uniform bounded
  weights, weight gain `1`.
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
operating-point-audit SHA-256 values. Freeze and obtain approval for a second
exact experiment plan before any optimizer probe or canary runs.

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
[Conv2 perfect-diode diagnostic card](results/index.md#perfectdiode-ordinary-mnist-lr-core-screen-20260727-conv2);
the failed-canary control is also bound by the
[approved Conv2 continuation plan](experiment_plans/perfectdiode-conv2-high-rho-corner-20260727-v1.md).
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

Conditional and ineligible stages publish explicit zero-work completions.
Manifests are immutable and embed upstream hashes. Resume skips only
hash-verified complete entries. An interrupted Adam candidate without a
complete optimizer-state checkpoint restarts from the shared initialization.

Required immutable evidence includes:

- approved experiment plan, canonical config, expanded manifest, resolved
  config, source commit/archive, effective-code fingerprint, and environment;
- train/validation split indices, T and K cohorts, training minibatch order,
  shared initialization checkpoint, and every parent-artifact hash;
- raw per-candidate T and K measurements, selectors, sentinels, and
  operating-point audit;
- optimizer probes, raw ordered LR vectors, canary/candidate results, complete
  step logs, epoch metrics, safety diagnostics, post-training T/K audits, and
  terminal surface handoffs;
- manifest and stage-completion hashes, plus an explicit
  `official_test_read=false` at every level.

Fail closed on missing approval, config/manifest drift, source or environment
drift, dataset/cohort/initializer mismatch, failed numerical smoke, failed
T/K gate, unsupported result-card schema, missing/non-finite artifacts, or
hash mismatch.

Terminal LR reasons include `selected_seed0_ordinary_mnist_three_epoch_screen`,
`unresolved_tk`, `unresolved_tk_gradient_viability`, `unresolved_probe`,
`unresolved_no_safe_center`, `unresolved_no_pass`, `unresolved_boundary`,
`unresolved_dead_gradient`, `unresolved_safety`, and
`unresolved_post_training_tk`. Every one of the six declared surfaces must
have one terminal selected-or-unresolved record.

Publish separate diagnostic comparison cards for T/K and the six-surface LR
screen through the existing result registry. No handoff status may imply
paper-facing evidence or a final-training learning rate.

## Local execution contract

This study may execute only in the existing `tmux main` and
`tmux akibscomputer` sessions. The routing is frozen:

| Scheme | T/K row and both optimizer surfaces |
|---|---|
| baseline `v1/c1` | `tmux main` |
| proposed/ours `v4/c1` | `tmux akibscomputer` |
| legacy `v4/c0.25` | `tmux main` |

Every cell of one scheme x optimizer surface stays on its assigned host. Its
scheme-specific T/K row runs on that same host. The initialization checkpoint,
dataset split, T/K cohorts, and training-order bytes must nevertheless be
hash-identical across both hosts.

Check both sessions and their GPUs immediately before every launch, use their
available capacity, run at most one worker per GPU, and never replace or
interfere with an occupied lane. If either lane is unavailable, wait. Do not
substitute Trex or Jean Zay.

Because scheme and host/runtime are confounded by this routing, the result may
select or reject an LR independently within each scheme x optimizer surface
but may not support a cross-scheme superiority claim. An Akib import, memory,
numerical-smoke, or scientific-preflight failure blocks the affected work and
requires an explicit amended plan. It must not trigger a silent host reroute,
batch-size reduction, or other scientific/configuration change.

Use the same immutable public launcher, resolved config, environment, device
type, output-writing path, and one-entry numerical smoke as production. Every
unique Conv3 scheme must also pass its exact T/K operating-point gate before
its LR surfaces are released. Any material source, config, environment,
initializer, architecture, nonlinearity, amplification, gain, or `T/K` change
invalidates the smoke and gate.

Before launch, create and validate two exact plans under
`docs/experiment_plans/`: one for T/K and, after T/K is hash-frozen, one for
LR. Each plan must bind its config and manifest SHA-256, job count, result
bundle, tracker entry, and planned comparison cards, and must record Filip's
explicit approval. This protocol alone is not launch authorization.
