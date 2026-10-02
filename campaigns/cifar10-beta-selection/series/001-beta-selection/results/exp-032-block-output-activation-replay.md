---
experiment: "exp-032"
evidence: "validated-local"
summary: "Initialization, float32, single case: linear analog outputs plus digital ReLU pass every layer at cosine ≥0.99 from 0.0025/0.035/0.0005 up to the top of the grid (1.0) in blocks1/2/3, versus diode windows of 0.005–0.1 / none / 0.001–0.125. Descriptive only; H-027 not graded"
verdicts: {"H-027": "inconclusive"}
---
# Result: at initialization, digital-ReLU outputs remove the upper beta limit

## Evidence and coverage

Scope reduced by Filip before completion: only `digital_relu-float32` ran. The
other five cases (diode and linear float32, all float64) are held in the queue, not
run. Root `results/cifar-block-output-init-replay-20261002-v1/`; bundle
`cells/digital_relu-float32/`, collected and validated locally
(`analysis/digital_relu-float32-collection.json`): 184 pooled rows, 256 images, all
targets within 0.33%, checkpoint/BN unchanged, PyTorch2.11.0+cu128. Ran on trex
RTX5090 shared with Ben and another user's job, 846s, peak 10.2GiB. trex rebooted
seconds after the runner and its validation exited 0, so the queue saw no terminal
receipt (job left in `attention`); the bundle was then collected manually with the
study's own `collect` step, without rerunning compute.

## Measurements

Contiguous run of targets with every layer's pooled local cosine ≥0.99, float32,
ours seed-0 initializer. Control: exp-030's float32 replay of the same checkpoint
(original diode outputs; no inactive channels at initialization, so its masked arm
equals the unmasked replay). The control grid stops at 0.4.

| Block | Diode outputs | Digital ReLU | Digital ReLU strict window |
|---|---|---|---|
| 1 | 0.005–0.1 | 0.0025–1.0 | 0.0025–1.0 |
| 2 | none (best 0.975 at 0.085) | 0.035–1.0 | 0.035–1.0 |
| 3 | 0.001–0.125 | 0.0005–1.0 | 0.001–1.0 |

At 1.0 the worst-layer pooled cosine is ≥0.9999 in every block. The float32 floor is
also lower in blocks 1 and 3. Targets 0.035–1.0 pass the strict rule (every minibatch
≥0.95, norm ratio 0.9–1.1) in all three blocks at once; the diode architecture has
no such common target, since block 2 never passes.

## Interpretation

With a linear analog output layer, the injected current lands on units that cannot
switch, and the finite-beta error that bounded the diode windows above about 0.1
does not appear even at a displacement equal to the free output RMS. Block 2's
diode failure at initialization, previously attributed to the float32 floor
(exp-030), is therefore mostly a nonlinearity effect: with linear outputs it
passes from 0.035. This is one architecture, one checkpoint and one precision, with
no `linear` or float64 case, so H-027's frozen rule is not applied (inconclusive).
Initialization has no dead block-3 channels; the trained case is
[exp-033](../experiments/exp-033-trained-linear-output-replay.md).

## Decision

Proceed to trained checkpoints (exp-033), float32 only, as Filip directed.
