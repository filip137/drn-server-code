# STAR-inspired hidden-state fault recovery in the IBM OM DRN

## Status and conclusion

This document records an `exploratory_noncanonical` fault-only experiment on
the IBM optimized-material (OM) DRN.  It tests the deliberately simplified
idea that a local representation target can be introduced through an
additional hidden-layer loss while retaining the repository's existing BPTT
and open-loop pulse-Adam machinery.

The main result is negative but informative:

- AIHWKit-compatible permanent corruption reduced the healthy 95.99% test
  accuracy to 34.43--59.31% across four independent fault masks.
- Output-only KL with one epoch of open-loop pulse Adam recovered 92.00% mean
  test accuracy.
- Adding per-example ReLU hidden KL produced 92.07%, only **+0.0675 percentage
  points** on average.  It won two masks and lost two masks.
- Healthy-DRN class-prototype KL and raw-state MSE produced 90.90% and 90.81%,
  respectively, both worse than output-only recovery on every mask.

The hidden objectives did change the representation they targeted.  The
per-example hidden-KL arm reduced its hidden KL by 74.9% relative to the
output-only arm, but that did not translate into a material classification
gain.  The simplest interpretation is that global output BPTT already
provides sufficient hidden credit assignment for this one-epoch recovery
problem.  A strong auxiliary representation constraint is largely redundant
and, for class prototypes, removes useful example-specific degrees of freedom.

This experiment is **not exact STAR and not local on-chip learning**.  The
hidden error is defined locally, but gradients are differentiated globally
through the reciprocal DRN equilibrium using BPTT; teacher activations and
Adam moments remain digital.

## Question

Starting from a damaged four-device DRN, does augmenting the ordinary output
KL objective with a hidden-state objective improve recovery from permanent
corrupt devices?

The experiment intentionally precedes a centered-equilibrium-propagation STAR
implementation.  It is a gradient-oracle/control for the objective idea:

\[
L = L_{\mathrm{output\ KL}} + \lambda_h L_{\mathrm{hidden}}.
\]

Four matched learning arms were evaluated:

1. `output_kl_only`: the existing ReLU-teacher output KL.
2. `relu_sample_hidden_kl`: KL to the frozen ReLU teacher's per-example
   50-dimensional hidden representation.
3. `healthy_class_hidden_kl`: KL to a label-indexed class mean from the healthy
   DRN.
4. `healthy_class_hidden_mse`: squared error to the healthy DRN's complete
   100-node class-mean hidden state.  This is the closest objective-level
   analogue to STAR's quadratic state repair term.

## Positive-conductance and fault contract

The implementation keeps AIHWKit's signed internal device coordinate and the
DRN circuit conductance explicitly separate:

\[
a \in [-1,1], \qquad G = a + 1 = 2x \in [0,2].
\]

No negative value of `a` is treated as a negative conductance.  Every circuit
evaluation receives the shifted, non-negative full conductance `G`.

The fault overlay follows the AIHWKit corrupt-device mechanism requested for
this study:

- each physical crosspoint is selected independently with probability
  `0.1348`;
- a corrupt raw coordinate is sampled uniformly from the intersection of the
  cell's own support and `[-0.01, +0.01]`;
- its lower and upper bounds collapse to that sampled value;
- both upward and downward pulse increments become zero;
- the learner receives only a blind pulse-write port and never receives the
  mask, cell bounds, verify values, or per-cell success information.

After the coordinate shift, corrupt cells therefore have
`G in approximately [0.99, 1.01]`, never negative conductance.  This differs
from the earlier provisional stuck-at-minimum/maximum proposal and is the
appropriate match to AIHWKit's corruption implementation.

The source population is the existing counterfactually repaired/Winsorized
87004 identity.  Fresh corruption is then overlaid explicitly.  This is a
fault-injection experiment; it must not be interpreted as AIHWKit enabling
published corruption by default, or as real hardware performing repair.

## Frozen initialization and experiment roles

The healthy DRN starts from the target-87004 ideal quantized mapping derived
directly from the frozen ReLU weights.  There is no program-and-verify step,
no QAT state, and no previous Adam-recovered state in the initializer.

The healthy identity gate reproduced:

| Split | Correct | Accuracy | Prediction SHA-256 |
|---|---:|---:|---|
| Validation | 4,776 / 5,000 | 95.52% | `8519adc9...e4aa15` |
| Test | 9,599 / 10,000 | 95.99% | `98ffa9a3...6644c` |

Fault seed 91501 was used only for development.  Seeds 91502--91505 were held
out until all hidden-loss settings were frozen.  These are four independent
fault masks on **one array identity**, not four arrays.

The realized evaluation-mask corruption counts were:

| Mask seed | Faulted cells | Fraction | W1 | W2 |
|---:|---:|---:|---:|---:|
| 91502 | 21,444 / 158,800 | 13.504% | 21,142 | 302 |
| 91503 | 21,353 / 158,800 | 13.446% | 21,081 | 272 |
| 91504 | 21,329 / 158,800 | 13.431% | 21,045 | 284 |
| 91505 | 21,584 / 158,800 | 13.592% | 21,334 | 250 |

## Objective calibration and selection

Healthy class prototypes were accumulated in float64 from exactly the 55,000
MNIST training examples.  Validation and test examples were excluded.  The
per-example ReLU hidden target used a separately fitted positive hidden gain
of 1.77828 and temperature 4.87954, the frozen teacher-hidden RMS on the 1,024
example calibration cohort.

Auxiliary weights were normalized by the ratio between output-only and
hidden-only W1 gradient RMS, averaged as mean batch RMS over 64 deterministic
batch-16 calibration steps.  Multipliers `{0.3, 1, 3}` were screened on fault
mask 91501.  Every hidden objective selected the smallest multiplier, 0.3:

| Arm | Gradient anchor | Selected hidden weight | Screen validation |
|---|---:|---:|---:|
| ReLU per-example hidden KL | 1.6131 | 0.4839 | 57.18% |
| Healthy class hidden KL | 3.5595 | 1.0679 | 57.10% |
| Healthy class raw-state MSE | 39.4955 | 11.8486 | 56.78% |

Selection at the lower grid boundary is evidence that the class objectives
were already too strong, not evidence that zero hidden weight was inferior.
The matched pulse run on the development mask gave 90.98% for output-only,
90.58% for per-example hidden KL, 89.14% for class KL, and 88.90% for class
MSE.  Pulse-count mismatch stayed below the declared 10% diagnostic threshold,
so no secondary output-only learning-rate control was triggered.

## Held-out result

### Accuracy by fault mask

| Mask | Faulted P0 | Output KL | + ReLU hidden KL | + class hidden KL | + class hidden MSE |
|---:|---:|---:|---:|---:|---:|
| 91502 | 44.21% | 92.06% | **92.37%** | 90.60% | 90.55% |
| 91503 | 34.43% | 92.14% | **92.23%** | 91.28% | 90.96% |
| 91504 | 59.31% | **92.12%** | 92.02% | 90.89% | 90.91% |
| 91505 | 47.11% | **91.69%** | 91.66% | 90.84% | 90.80% |
| **Mean** | **46.27%** | **92.0025%** | **92.0700%** | **90.9025%** | **90.8050%** |

Paired mean changes versus output-only were:

- ReLU per-example hidden KL: **+0.0675 pp**, with 2/4 wins.
- Healthy class hidden KL: **-1.1000 pp**, with 0/4 wins.
- Healthy class raw-state MSE: **-1.1975 pp**, with 0/4 wins.

Relative to the 95.99% healthy reference, output-only recovered 91.70% of the
fault-induced accuracy loss on average.  Per-example hidden KL recovered
91.80%.  The approximately 0.10 percentage-point difference in recovered
damage is not a meaningful practical improvement at this coverage.

The predeclared success criterion required at least +2 pp mean accuracy and
wins on at least three of four masks.  No hidden arm passed it.  The
per-example arm did pass the representation criterion: its mean target hidden
KL fell from 4.6041 under output-only recovery to 1.1564, a 74.9% reduction.
Class KL fell 59.9% and class MSE fell 48.9%, but their mean output KL worsened
from 0.1874 to approximately 0.227 and their classification accuracy fell.

This separation is important: the optimizer successfully satisfies each
auxiliary loss, yet those healthier-looking representations do not yield a
better damaged-network decision function.

## Pulse and fault-safety audit

All recovery arms used one epoch of direct physical open-loop pulse Adam with
the same post-fault persistent state, minibatch order, and pulse-selection RNG
within each mask.  There was no program-and-verify operation.

| Arm | Mean requested pulses | Relative to output-only | Mean suppressed at faults |
|---|---:|---:|---:|
| Output KL | 55,143 | baseline | 7,463 |
| ReLU hidden KL | 60,254 | +9.27% | 8,205 |
| Class hidden KL | 57,343 | +3.99% | 7,708 |
| Class hidden MSE | 60,401 | +9.53% | 8,236 |

Across all 16 held-out recovery arms:

- verify reads: exactly zero;
- corrupt-cell movement: exactly zero;
- pulse-cap events: zero;
- probability-clipping events: zero;
- inference-read, cycle-to-cycle, and apparent-write noise: disabled by the
  fault-only protocol.

Stochasticity in this rung is the Bernoulli pulse-command realization.  The
device response itself is deterministic but retains each healthy cell's
heterogeneous sampled bounds and pulse increments.  Programming error and
program-and-verify observation error were intentionally excluded.

## Interpretation

### What the result supports

Permanent faults are severe in this four-device DRN: roughly 13.5% corrupt
physical cells affect a much larger fraction of logical four-cell synapses,
and the untrained damaged network falls to 34--59%.  Nevertheless, the
remaining healthy conductances provide enough degrees of freedom for
target-specific pulse training to recover about 92%.

The ordinary output loss already sends a global BPTT signal through the
hidden equilibrium.  The per-example hidden KL therefore contributes mostly
redundant representation pressure.  It greatly improves the requested hidden
metric, costs about 9% more pulse attempts, and produces only a +0.07 pp mean
accuracy change.

Class prototypes are a worse target for this network and budget.  They are
attractive from an SRAM/locality perspective, but collapsing every example of
a class toward one healthy mean removes within-class structure.  The damaged
network may also need a different internal representation from the healthy
network because some physical degrees of freedom are permanently unavailable.
Forcing the original representation can therefore compete with finding a new
task-effective conductance state.

### What the result does not establish

This does not reject the STAR algorithm.  Exact STAR changes both the learning
rule and the hardware information path: centered positive/negative phases and
local squared-voltage-drop differences replace BPTT and digital Adam.  The
present experiment only rejects the stronger claim that adding the tested
hidden objectives to the current BPTT recovery path materially improves
accuracy.

It also does not establish fabricated-device behavior.  The array is an
analyst-Winsorized, counterfactually repaired AIHWKit OM identity with an
explicit synthetic AIHWKit-compatible corruption overlay.  It uses one array
identity, four evaluation masks, one data/model seed, one epoch, and no
programming/read/retention noise.

## Recommended next step

The result does not justify a larger hidden-KL sweep as the main research
direction.  If one cheap follow-up is desired, test a smaller per-example
hidden-KL grid `{0.03, 0.1, 0.3}` because 0.3 won at the current lower boundary;
this would determine whether the tiny positive mean was regularization or
noise.  It should remain a BPTT feature-distillation control.

The scientifically distinct next step is the actual centered-EP STAR arm:

- use the same frozen healthy initializer and exact post-fault P0 clones;
- retain AIHWKit-compatible corrupt values and positive `G=a+1` conductances;
- compare task-only centered EP, paper-exact quadratic repair, and STAR-CF;
- generate physical pulse updates from local positive/negative squared voltage
  differences;
- keep convergence, residual, voltage, and saturation gates;
- retain output-only BPTT+Adam as a privileged upper control, not as on-chip
  evidence.

## Reproducibility

- Branch: `codex/ibm-om-star-hidden-kd-fault-recovery`
- Implementation commit: `a09a018d9f94f7034242cb3a71f550abbb18e441`
- Run ID: `20260831T141052.896675Z-acc2cb10-219d8852`
- Runtime: 835.64 seconds on local RTX 3090
- Result root:
  `results/mnist-ibm-om-star-inspired-hidden-kd-fault-recovery-exploratory-20260831-v1/`
- Registered artifacts: 29; all paths, sizes, and SHA-256 digests verified
- Focused tests: 25 passed

The run manifest records the committed numerical implementation above, but
also records `dirty=true` with dirty hash
`c7debc34bb5139806c7a9d770eeeedb303f3ae551d6e0916fcdf44e29c99e7a1`
because three pre-existing user documentation files remained modified.  The
result is therefore correctly retained as exploratory, not promoted to
canonical evidence.
