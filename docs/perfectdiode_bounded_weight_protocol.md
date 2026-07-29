# Perfect-Diode Bounded-Weight Initializer And Rho Protocol

Updated: 2026-07-29

Status: active scientific specification; all-depth execution and global
initializer selection are pending.

## Purpose

Select one bounded initialization method and complete bounded
parameter-specific learning-rate handoffs before deterministic medium-affine
paper training.

This is an ordinary-MNIST optimization study. It covers all 18 surfaces:

```text
3 architectures x 3 amplification schemes x 2 optimizers
```

The architectures, schemes, BPTT algorithm, perfect-diode model, output, loss,
and optimizers are those in the
[paper experiment definition](conv_paper_experiment_definition.md).

## Dataset And Shared Assets

Use ordinary MNIST with the deterministic stratified 55,000/5,000
training/validation split and seed `0`. Do not instantiate or read the
official test split.

For every architecture, materialize the split, diagnostic cohorts, training
order, and one initialization checkpoint independently for each initializer.
Within one architecture and initializer, share the exact checkpoint across
schemes and optimizers wherever tensor geometry agrees.

The two initializer families must use identical model seed, loader seed,
cohorts, and minibatch bytes. Reset global layer and parameter counters before
every independent build.

## Bounded Weight Contract

Only `ConvWeight_*` and `DenseWeight_*` are conductance-bounded:

```text
training projection interval = [1e-5,1e-4]
```

Biases retain the active perfect-diode bias initialization and LR policy and
are not projected into this interval.

Test both initializers:

### Full-range bounded uniform

```text
weight_initialization = bounded_uniform
G ~ Uniform[1e-5,1e-4)
```

The draw spans the full physical interval. `weight_gain` does not change this
distribution.

### Bounded Kaiming uniform

```text
weight_initialization = bounded_kaiming_uniform
weight_gain = 1
m = (lower + upper) / 2
h = (upper - lower) / 2
G = m + h * U[-sqrt(6/fan_in),sqrt(6/fan_in)]
```

The implementation samples below the upper support endpoint. Training
projection is practically closed, so an optimizer step may project a weight
to exactly `1e-4`.

The immutable
[`perfectdiode_conv12_initialization_rho_comparison_20260727_v1.json`](../configs/conv/perfectdiode_conv12_initialization_rho_comparison_20260727_v1.json)
records an earlier Conv1/Conv2 design. Preserve it unchanged as provenance; it
does not complete this all-depth selector.

## Operating Points

Import each surface's resolved wide-range input gain and operational `T/K`.
Do not recalibrate or silently change them for an initializer.

Before rho probing:

- Conv1/Conv2 repeat their fixed-`T/K` gradient security comparison under each
  bounded initializer; and
- Conv3 repeats its operating-point audit and reference-gradient viability
  gates under each bounded initializer.

A failure leaves that initializer/surface unresolved. It does not trigger a
new `T/K`, cross-initializer transfer, or fallback.

## Independent Rho Selection

Run rho selection independently for every initializer x architecture x scheme
x optimizer surface:

- recompute optimizer-specific nominal-LR-one proposal units;
- derive a fresh raw LR for every Conv weight, Dense weight, and bias;
- apply the architecture's active core-grid and expansion rules from
  [`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md) or
  [`perfectdiode_conv3_learning_protocol.md`](perfectdiode_conv3_learning_protocol.md);
- restart every 640-step canary and three-epoch candidate from that
  initializer's shared checkpoint; and
- perform the protocol's terminal selection and post-training `T/K` checks.

Never reuse a wide-range probe, raw LR, candidate, or selected rho result.
Never transfer numerical outputs between bounded initializers.

Every surface must produce one selected safe LR handoff or an explicit
unresolved reason.

For the active baseline/ours-only focused study, `90%` is a reporting
threshold rather than a condition for withholding a rho handoff. If no
safety-clean candidate reaches it, select the best completed candidate with
the normal loss-plateau and tie-break rules, and report whether that selected
rho is on a tested outer bound. A bound-selected rho triggers the single
factor-of-three expansion wave.

The same wave may also be triggered when the best safety-clean core accuracy
is below the declared suspicious-accuracy floor (`80%` in the active
configs), even if the selected rho is interior. For each otherwise interior
axis, extend toward the outer core edge with the higher safety-clean
validation accuracy. Do not infer a direction from rejected cells. This
remains one wave with at most one added value per axis and at most 16 tested
cells. Record whether the maximum accuracy remains below the diagnostic floor
after expansion. Bound occupancy and projection efficiency remain
report-only.

## Global Initializer Selection

An initializer is globally eligible only when all 18 surfaces have a selected
safe LR handoff after the permitted expansion and post-training checks.

If exactly one initializer is eligible, select it. If neither is eligible,
leave the bounded paper condition unresolved and run no bounded paper rows.

When both are eligible, compare their matched selected candidates for every
surface. For losses `L_uniform` and `L_kaiming`:

```text
surface tie when max(L_uniform,L_kaiming) / min(L_uniform,L_kaiming) <= 1.02
otherwise the lower-loss initializer wins the surface
```

Select the initializer with more surface wins. Tied surface counts do not
count as wins.

Break an equal win count by:

1. lower median over surfaces of
   `L_initializer / min(L_uniform,L_kaiming)`;
2. higher median final validation accuracy; then
3. `bounded_kaiming_uniform`.

Publish the complete paired table, win/tie counts, tie-break statistics,
selected initializer, and all source hashes. No subjective override or
architecture-specific initializer is allowed.

## Paper Handoff

For the selected global initializer, freeze all 18 ordered raw LR vectors.
The corresponding deterministic medium-affine paper rows use those vectors
unchanged and start from fresh paper-run initializations under the selected
bounded initializer.

The bounded result is a combined physical-range and initialization contract.
If full-range uniform wins, do not describe the wide-range-versus-bounded
comparison as changing only the conductance bounds.

Every handoff and selector artifact records `official_test_read=false`.
Ordinary-MNIST metrics select the setup but are not paper results.
