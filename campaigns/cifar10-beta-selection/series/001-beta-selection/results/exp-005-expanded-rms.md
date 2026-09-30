---
experiment: "exp-005"
evidence: "validated-local"
summary: "Expanded RMS0.0005–1.5 brackets all sampled peaks. Block2 prefers larger RMS than blocks1/3; block3 tolerates RMS≈0.40. R≈0.075 still transfers across schemes."
verdicts: {"H-002": "supports", "H-003": "supports"}
---
# Expanded RMS search

All three schemes completed22 targets x8 Conv weights x8 minibatches:176 pooled
and1408 batch rows,66 calibrated block targets each. Local bundles and coverage
validate; all cosines are defined, model/BN tensors unchanged, and calibration
and gradient-replay RMS agree. Maximum target errors: baseline2.78%, ours2.78%,
legacy3.00% (below3%). No failures or exclusions. Charged5090 time: baseline281.0s,
ours324.2s, legacy321.1s;926.4s total, within3600s budget. GPUs released.

Same initialization/cohort/native endpoint convention and fixedT/K as exp-004.
Combine new points with exp-003/004, using new measurements at overlapping nominal
targets0.02/0.075. Their maximum pooled cosine change is0.000422; beta was
recalibrated within3%, so these are overlapping targets, not exact-beta replays.

Best sampled **relative RMS**, maximizing the minimum pooled native-BPTT cosine
among the Conv weights in each block; parentheses give that minimum cosine:

| Block | Conv layers | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| 1 | 3 | 0.0140 (0.99945) | 0.0348 (0.99935) | 0.0140 (0.99934) |
| 2 | 3 | 0.0448 (0.99320) | 0.0821 (0.96839) | 0.0647 (0.97301) |
| 3 | 2 | 0.0100 (0.99969) | 0.0100 (0.99961) | 0.0140 (0.99944) |

All peaks are interior to the expanded range, but blocks1/3 have broad plateaus.
For ours, sampled points within0.001 cosine of the peak span roughly0.010–0.065
in block1 and0.004–0.035 in block3. These describe flatness, not confidence bounds.
At the smallest RMS≈0.0005, block2 cosine falls to0.243 in ours and0.225 in legacy;
shrinking the perturbation indefinitely does not improve this fixed-T/K estimator.
The cause of this low-displacement degradation is not isolated here.

Largest **tested** RMS with every pooled Conv cosine>=0.95:

| Block | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| 1 | 0.162 | 0.136 | 0.133 |
| 2 | 0.398 | 0.111 | 0.162 |
| 3 | 0.397 | 0.397 | 0.398 |

These are sampled passing values, not exact cutoffs. The two-convolution block
tolerates more displacement than the other blocks in ours/legacy, but baseline's
three-convolution block2 tolerates just as much. Blocks1/2 have the same depth
and different peaks: depth alone is not a sufficient predictor. Position, width,
spatial resolution and signal statistics remain confounded.

R≈0.075 remains a shared candidate: worst measured minibatch/layer cosine is
0.9710 baseline,0.9789 ours,0.9607 legacy. H-002/H-003 remain supported within this
initialization scope. Expanded pooled shared>=0.95 span is approximately0.035–0.111;
no shared>=0.99 span exists. Finer sampling reveals a legacy block2 minibatch dip
to0.9442 nearR0.055: the previous0.05–0.09 all-batch span was not a certified
continuous interval. The connected sampled all-batch overlap is now roughly
0.060–0.091, plus the isolated passing point near0.05. Interpolation between
tested points remains unverified. Matched-local references preserve the broad picture.

Decision: retain0.075 as a robust shared training candidate; the per-block peaks
above are alternative gradient-alignment candidates, not established training optima.
No training was launched. Exact peak betas and sampled bands are in
`results/cifar-block-rms-expanded-20260929-v1/analysis/peaks.json` and `comparison.json`.

[Expanded sweep figure](../../../../../results/cifar-block-rms-expanded-20260929-v1/analysis/scheme_rms_cosine_expanded.jpg).
Raw and validated summaries: each scheme's `run/` and `analysis/` beneath that root.
