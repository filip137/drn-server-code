---
experiment: "exp-009"
evidence: "validated-local"
summary: "Fixed initialization beta yields much smaller trained displacements and poor block2 alignment in all schemes; larger initial RMS0.09 helps but does not preserve alignment."
verdicts: {"H-007": "supports"}
---
# Fixed initialization beta loses effective displacement

All nine checkpoint replays and collections completed: two fixed beta vectors,
144 pooled Conv-weight rows and 1152 batch rows, same 256 training examples as
prior diagnostics. Epoch-zero evidence reused with exact coefficient provenance.
No undefined cosines, nonfinite failures, exclusions or optimizer updates;
checkpoint bytes and model/BN tensors unchanged. Native/local reference cosine
is at least 0.9999986. Charged runtime 436.84 seconds within 2700; maximum reserved
CUDA memory 10.49 GiB. Queue supervision and GPU reservations have ended.
Four unused placement submissions were cancelled without attempts when Trex
became occupied and the remaining work was redistributed; no compute repeated.

Worst pooled Conv-weight cosine across all three blocks (block2 is weakest in
every trained case):

| Initial RMS target | Scheme | Epoch0 | Epoch10 | Epoch30 | Epoch50 |
|---|---|---:|---:|---:|---:|
| 0.03 | Ours | 0.9450 | -0.0080 | 0.0766 | 0.0326 |
| 0.03 | Baseline | 0.9904 | 0.1106 | 0.1923 | 0.0921 |
| 0.03 | Legacy | 0.9492 | 0.0679 | 0.1005 | 0.1139 |
| 0.09 | Ours | 0.9674 | 0.0346 | 0.0849 | 0.0900 |
| 0.09 | Baseline | 0.9913 | 0.1913 | 0.2482 | 0.1385 |
| 0.09 | Legacy | 0.9690 | 0.1105 | 0.1783 | 0.1908 |

All 54 trained scheme/block/setting displacements are smaller than at
initialization. For initial RMS0.09, block2 RMS shrinks roughly 650–37,000-fold.
Example, ours has actual RMS vectors:

| Epoch | Block1 | Block2 | Block3 |
|---|---:|---:|---:|
| 0 | 0.0908 | 0.09085 | 0.09076 |
| 10 | 0.001191 | 0.000009983 | 0.00001767 |
| 30 | 0.004738 | 0.00005672 | 0.0002341 |
| 50 | 0.001982 | 0.00001870 | 0.0001193 |

The larger beta improves pooled alignment in 25/27 trained block comparisons;
exceptions are legacy block1 at epochs30/50. Nevertheless, only 9/54 trained
block/settings exceed 0.95, and neither vector keeps all blocks well aligned at
any trained checkpoint. The earlier constant-RMS sweeps recalibrated beta and
therefore answer a different question.

H-007 is supported in this scope: fixed initialization beta does not preserve
effective displacement or gradient quality on these BPTT trajectories. This
motivates adjusting beta during training, while retaining the earlier finding
that a high fixed RMS target can also distort gradients. The observed drift
combines changes in states, conductances and loss-gradient forcing; this replay
does not isolate intrinsic susceptibility. Small-displacement numerical effects
are a possible contributor, not a demonstrated sole cause. No read noise was
injected, and no EqProp training accuracy was measured.

The six requested fresh 10-epoch training arms remain recorded in exp-002 and
have not been launched. Their implementation must explicitly distinguish frozen
initialization beta from recalibration to maintain RMS.

Artifacts: `results/cifar-block-fixed-beta-20260929-v1/analysis/` contains exact
initialization provenance, validated per-checkpoint summaries, `comparison.json`,
`points.csv`, `queue_completion.json` and reproducible `analyze.py`.
Plots: [initial RMS0.09](../../../../../results/cifar-block-fixed-beta-20260929-v1/analysis/fixed_beta_initial_rms_0.09.jpg)
and [initial RMS0.03](../../../../../results/cifar-block-fixed-beta-20260929-v1/analysis/fixed_beta_initial_rms_0.03.jpg).
