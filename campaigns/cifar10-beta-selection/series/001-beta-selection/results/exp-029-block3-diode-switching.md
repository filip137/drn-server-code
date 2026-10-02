---
experiment: "exp-029"
evidence: "validated-local"
summary: "Fixing diode decisions restores block3's first failing local EP gradient at ours epoch1"
verdicts: {"H-024": "supports"}
---
# Block3's upper RMS limit is sensitive to diode switching

At the first failing beta, holding diode decisions to the zero-nudge trajectory
raises worst minibatch/layer cosine from **0.924610 to0.999984**, with norm
ratios0.997295–1.000734. This supports switching as a cause of this checkpoint's
small upper displacement limit. The input, output-loss forcing, weights, beta,
K4 and block-local BPTT reference were unchanged. This is an artificial
mechanism intervention, not a proposed training solver.

## Evidence and coverage

[exp-029](../experiments/exp-029-block3-diode-switching.md) completed all five
native targets, the fixed-decision low-beta control and the first-failure
intervention:256 training images, eight batches of32, two layers of block3,
ours EqProp epoch1, full float64 including free capture. Free T=[6,6,4], local
BPTT K4 and EP K4. All112 layer/minibatch gradient comparisons are retained;
the BPTT reference was computed once per batch. The smallest target passes
the prospective cosine/norm rule. Each pooled output target matched within3%.

Bundle: `results/cifar-block3-switching-20261001-v1/cells/eqprop-ours-e1/`.
`artifacts/comparison.json` contains layer/batch measurements and per-update
switch/margin summaries; `reference.pt` preserves zero masks, states and BPTT
gradients. `analysis/summary.json` and
[JPG figure](../../../../../results/cifar-block3-switching-20261001-v1/analysis/block3_switching.jpg)
summarize the evidence. Frozen source/config/commands and checkpoint provenance
are linked by the experiment handoff and bundle manifest. Checkpoint SHA256:
`6910a9f414acf7c846dddd1b56d8bf1f7bd912c386c652cf5c93a68d5c53e5bd`.

Canonical and diagnostic validation passed remotely and after collection;
source/input bytes and parameters/BN buffers remained unchanged. Four focused
CPU tests passed, including exact tracing/EP equivalence and a complete tiny
256-image replay. PyTorch2.11.0+cu128/CUDA12.8 on Loulou5090; total charged
runtime379.71s of1200s, peak reserved20.36GiB. Initial Fifi placement was
cancelled before any attempt because another queued job occupied the GPU.

## Measurements

Cosines below are minima over the eight minibatches, separately for each layer.
RMS is the pooled centered output response divided by saved free output RMS.

| Target output RMS | Achieved RMS | Beta | Layer1 cosine | Layer2 cosine |
|---|---:|---:|---:|---:|
| 1e-6 | 1.00001e-6 | 0.164560 | 0.999984 | 0.999998 |
| 1e-5 | 9.99984e-6 | 1.645598 | 0.999986 | 0.999977 |
| 2e-5 | 2.01647e-5 | 3.291196 | 0.999902 | 0.996775 |
| 3e-5 | 2.97696e-5 | 4.294787 | 0.998292 | **0.924610** |
| 1e-4 | 9.70126e-5 | 6.742294 | 0.970779 | **0.443886** |
| Fixed decisions, same beta as3e-5 | 2.60988e-5 | 4.294787 | **0.999984** | **1.000000** |

The intervention keeps beta fixed, so its achieved RMS can change; it is not
recalibrated to3e-5. All batches/layers pass cosine>0.95 and norm0.9–1.1
through2e-5. At1e-4 the worst norm ratio reaches2.24551. The earlier1e-4
cosine0.43006 used a slightly different calibrated beta6.85814; the tiny-target
and3e-5 measurements reproduce the earlier results closely.

At3e-5, only **5 layer1 and512 layer2 diode-decision events** differ from
zero nudge, among134,217,728 update entries per layer across both phases.
These are events, not unique neurons. Fixed-mask artifact switch counts report
the candidates' proposed decisions; the intervention overrides those decisions.
The pooled response's deviation from small-beta linearity is14.33% in layer1
and54.88% in layer2. Phase asymmetry is14.26% and48.11% of the centered
response norm. Fixed decisions reduce these defects to approximately1e-8
or below while restoring gradient agreement with the independently held BPTT.

Margins were measured before clipping. Across the zero-nudge update entries,
minimum absolute margins are3.59e-8 and1.95e-10 for layers1/2;32 and47 entries
respectively lie within1e-6 of their boundary. Global RMS does not describe
these rare near-boundary candidates.

Zero-nudge continuation drifts by relative RMS2.24e-4 in layer1 and7.29e-4
in layer2, larger than the tested centered responses. This common drift also
exists at the passing small beta and during the successful fixed-decision
replay. Its magnitude alone therefore does not explain the gradient failure;
interactions between settling and switching are not isolated by this test.

## Interpretation and decision

H024 is supported for ours EqProp epoch1, block3, finite K4. A tiny fraction
of changing diode decisions can materially distort the local gradient despite
tiny average output displacement. The fixed-reference gradient recovery is
the causal evidence; affine responses under fixed decisions alone would not
establish it. This does not identify which individual switches dominate, prove
a universal threshold, or explain block2 and other checkpoints automatically.

Keep1e-5 as the existing conservative cross-checkpoint block3 target. This
single checkpoint passes2e-5, but that does not establish a replacement for
the common target. A useful extension would isolate which layer's switching
drives the failure, or repeat the intervention in block2. No extra cases were
added to this bounded experiment.
