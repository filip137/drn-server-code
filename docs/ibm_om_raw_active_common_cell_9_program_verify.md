# IBM OM raw-active p90 common-cell nine-level P&V

- Status: implemented characterization protocol
- Device: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Evidence class: model-based hardware-derived fitted preset
- Scope: physical-cell programming only; no QAT or on-chip update

## Question

Can one array-wide baseline and one array-wide increment define nine physical
conductance targets that remain programmable across the IBM OM cell
population when each DRN conductance is represented by the single active
state `a_i`?

This is different from the earlier `raw_active_p90_quad` study.  That study
used one global logical differential range but selected a different baseline
for every four-cell quad.  Here neither the baseline nor the cell increment
may vary by cell, pair, quad, layer, assignment, or target.

## Physical coordinate

Every trajectory evolves the single programmable active state `a_i`.  The
AIHWKit reference `r_i` is retained as identity provenance but is excluded
from pulse dynamics, bounds, target construction, verify values, controller
decisions, and analysis.

The frozen array-wide display/controller coordinate remains

```text
A_min = -3.455834150314331
S     =  5.994294881820679
g     = (a - A_min) / S.
```

There is no cell-wise normalization.

## Global p90 cell interval and nine targets

On the repaired development assignment 84001, the widest exact interval
contained in at least 90% of all 158,800 individual cell supports is

```text
[0.5506438854350477, 0.6496889454832263].
```

The pulse plant is float32.  The executable endpoints are therefore rounded
one float32 value inward whenever ordinary rounding would leave the exact
interval:

```text
g_min,exec = 0.5506439208984375
g_max,exec = 0.6496888995170593.
```

This preserves exactly 142,920/158,800 development cells.  Define

```text
B       = (g_min,exec + g_max,exec) / 2
        = 0.6001664102077484
Delta_g = (g_max,exec - g_min,exec) / 8
        = 0.012380622327327728
g_k     = B + (k - 4) Delta_g,  k=0,...,8.
```

Thus the nine states are cell states, not nine logical four-cell
differentials.  No logical-weight mapper is chosen by this experiment.

The same frozen codebook is applied without adaptation to held-out assignment
85001.  Its hidden-bound audit is analysis-only; it must not alter targets or
controller decisions.

## Program-and-verify

Each device/repeat is initialized at raw `a=0` and conditioned toward its
RESET boundary until four consecutive persistent changes are below `2e-6`,
or until 4096 conditioning pulses.  One immutable conditioned state is cloned
across all nine targets.

Target programming uses the existing one-pulse controller:

```text
if abs(g_apparent - g_target) <= tau: accept
elif g_apparent < g_target - tau:     apply one SET pulse
else:                                apply one RESET pulse
```

with

```text
tau = 0.00791586015294392
maximum target-programming pulses = 128.
```

Conditioning and target programming have independent per-trajectory seeds.
RESET pulses after overshoot remain allowed.  The controller receives only
its target, apparent verify values, and its own history.

## Acceptance is not distinguishability

For this codebook,

```text
Delta_g / 2 = 0.006190311163663864
2 tau       = 0.01583172030588784 > Delta_g.
```

Adjacent apparent acceptance windows overlap.  Therefore the study must
report all of the following separately:

1. exact hidden target support;
2. verify-window intersection with support;
3. apparent controller acceptance;
4. persistent-within-`tau` after acceptance;
5. persistent nearest-code correctness;
6. persistent error no larger than half a code step;
7. endpoints compatible with more than one acceptance window; and
8. complete identity/repeats for which all nine targets meet each criterion.

An accepted endpoint must not be called a distinguishable nine-state result
unless it also passes the declared nearest-code or half-spacing diagnostic.

## Evaluation populations

The production arms use all 158,800 cells for exact support auditing and a
frozen uniform sample of 1,024 physical identities for pulse execution.  Each
identity has four independent conditioning/programming repeats at all nine
targets, for 36,864 trajectories per assignment.  Identity partitions are
fixed before execution; the validation partition owns the primary
programming metrics.

This characterization is a prerequisite for, not part of, a later QAT
experiment.  A QAT mapper may be declared only after the nine-cell-state
programming result is reviewed.

