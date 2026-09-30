---
experiment: "exp-023"
evidence: "validated-local"
summary: "BN keeps block2/3 injected RMS near 90–100 despite large internal voltage changes; head input RMS falls from 96.7 to 50.9."
verdicts: {}
---
# Result: post-BN and injected voltages at epochs 0/1/5

## Evidence and coverage

Root: `results/cifar-boundary-voltage-20260930-v1/`. All 3×256 training-example
forward replays completed, with 84 pooled stage rows, 672 batch-stage rows,
72 BN batch records and per-channel statistics. Same ours fixed-0.3× EqProp
trajectory, cohort order, batch32, finite T=[6,6,4], V4/C1, float32 and
torch2.11.0+cu128/CUDA12.8 as exp-022. BN uses training minibatch statistics
with running buffers frozen; these are not inference-mode running-stat outputs.

The local two-example CPU smoke and collected GPU bundle passed canonical
validation. The 24 internal-state RMS values matched exp-022 to maximum
relative error 2.22e-16. Input/source/runtime/restore identities, parameters,
BN buffers and checkpoint bytes were checked. Every observed physical boundary
exactly equaled `[g*x, -g*x]`; logical inputs equaled the captured preceding BN
outputs. Double-precision BN formula reconstruction maximum absolute error
was 2.40e-6. Independent count/sum/square pooling, affine RMS and gain RMS
identities passed. No failed or excluded scientific cases. Runtime 39.61s;
peak reserved CUDA memory 1.19GiB. Original checkpoints remain under exp-022's
input root; exact hashes and checkpoint paths are in the replay manifest/config.

Measured source identity: `4e9cd31010212c3e835c7813216e796f5ae8bb43f79f80451fb6271e07404443`.
Voltage artifact SHA256: `de7283a2a1bc7f76df22a3e54d000c3c05e32516adf00da2baf367ec9ad9876e`.

## Measurements

All numbers below are RMS over the identical 256-example cohort. The pre-BN
tensor is the max-pooled differential block output, distinct from the raw
physical state used in exp-022's earlier voltage plots.

| Boundary / stage | Epoch 0 | Epoch 1 | Epoch 5 |
|---|---:|---:|---:|
| Block1 before BN | 0.021014 | 0.225016 | 0.042098 |
| Block1 after BN + affine | 0.974635 | 0.987448 | 0.967117 |
| Physical input to block2 | 97.4635 | 98.7446 | 96.7115 |
| Block2 before BN | 0.008773 | 6.253581 | 2.348446 |
| Block2 after BN + affine | 0.882826 | 0.963355 | 0.919829 |
| Physical input to block3 | 88.2826 | 96.3350 | 91.9789 |
| Block3 before BN | 0.022497 | 3.748302 | 0.837363 |
| Block3 after BN + affine | 0.966513 | 0.838255 | 0.509775 |
| Physical input to dense head | 96.6513 | 83.7851 | 50.9099 |

Block1's physical image-input RMS is 40.6908, 40.6911 and 40.6915. Input gains
remain essentially 100: block2 100→99.9997→99.9997; block3
100→99.9996→99.9956; head 100→99.9518→99.8673.

Signed post-BN means (0/1/5): block1 approximately 0/−0.1411/−0.1802;
block2 approximately 0/−0.1467/−0.0921; block3 approximately
0/+0.0455/+0.0553. These are learned affine shifts. Physical signed means
cancel because each logical voltage is duplicated with its negative.

The post-BN RMS obeys the measured channel/batch identity
`RMS² = mean_channels,batches(gamma² * variance/(variance+epsilon) + beta²)`.
All epsilon values are 1e-5. Thus even initialization is below unit RMS:
block2's normalized RMS is 0.8828 before and after its initially identity
affine transform. At epoch5, block3's normalized-before-affine RMS remains
0.9712, but gamma RMS has fallen from 1 to 0.5229 and post-affine RMS is
0.5098. Head gain changes by only 0.13%; it is not the main source of the
head-input reduction.

Plots and tables under `analysis/`:

- [RMS before BN, after BN and injected into the next stage](../../../../../results/cifar-boundary-voltage-20260930-v1/analysis/boundary_voltage_rms.jpg)
- [Signed logical means](../../../../../results/cifar-boundary-voltage-20260930-v1/analysis/boundary_voltage_means.jpg)
- `voltage_epoch_comparison.csv`: all 84 stages, mean/RMS/std/min/max/zero fraction.
- `channel_voltages.csv`: per-channel pre-BN, normalized and post-affine values.
- `boundary_summary.csv` and `summary.json`: nine boundaries/epochs and validation.

## Interpretation and decision

BN removes most of the upstream RMS-scale growth before the next block.
At epoch1 the pooled block2 output RMS rises 712.8×, but the physical input
to block3 rises only 1.091×. Block2's injected input stays within 1.4% of its
initial RMS across these checkpoints. The head input instead falls 47.3%
by epoch5, predominantly through the learned block3 BN affine scale.

This does not mean BN preserves the input distribution or channelwise means,
and it does not normalize each convolution inside a block. The large changes
in raw internal state RMS therefore coexist with fairly stable block2/3 input
RMS. Absolute state scale alone remains insufficient to establish what limits
early-layer gradient agreement; the present replay measures boundary voltages,
not a causal intervention on displacement, parameters or distributions.

The requested descriptive comparison is complete. No intermediate epochs
beyond 0, 1 and 5, alternative trajectories or training follow-up are claimed.
