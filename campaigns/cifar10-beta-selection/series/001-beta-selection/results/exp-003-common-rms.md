---
experiment: "exp-003"
evidence: "validated-local"
summary: "A common relative output RMS band around 0.04–0.11 yields pooled cosine >=0.95 across all CIFAR blocks; minibatch agreement narrows the shared band to roughly 0.04–0.09."
verdicts: {"H-002": "supports"}
---
# Result: common output displacement and gradient cosine

Coverage: 12 focused levels plus the prior 0.01x reference, all three blocks / eight Conv tensors / eight batches (104 pooled and 832 batch rows). Loulou RTX5090 compute 200.8s; peak reserved 10.21GiB. Same original initializer, 256 training images, batch32, native T/K=[6,6,4], frozen summed-CE force and no noise. Model/BN unchanged. Local bundle, coverage and previous-reference agreement validated. No failed or excluded cases. Source/config/input hashes are retained in the canonical bundle.

B samples were placed using the previous response curve; their newly measured relative RMS values differ from nominal targets by at most 1.13%. The table labels nominal targets for readability; the plots and interval calculations use actual RMS. R means RMS((s_plus-s_minus)/2)/RMS(s_free).

Minimum native-BPTT cosine across the Conv weights in each block, for the cohort-averaged gradients:

| Nominal R | Block 1 | Block 2 | Block 3 |
|---:|---:|---:|---:|
| 0.02 | 0.9993 | 0.9058 | 0.9993 |
| 0.03 | 0.9993 | 0.9450 | 0.9990 |
| 0.04 | 0.9991 | 0.9614 | 0.9983 |
| 0.05 | 0.9990 | 0.9603 | 0.9976 |
| 0.06 | 0.9987 | 0.9522 | 0.9969 |
| 0.075 | 0.9979 | 0.9648 | 0.9964 |
| 0.09 | 0.9955 | 0.9674 | 0.9958 |
| 0.11 | 0.9840 | 0.9593 | 0.9941 |
| 0.135 | 0.9517 | 0.9403 | 0.9918 |
| 0.165 | 0.8883 | 0.9435 | 0.9885 |
| 0.2 | 0.8116 | 0.9464 | 0.9843 |
| 0.25 | 0.7418 | 0.9003 | 0.9778 |

**Common band:** sampled pooled-cosine >=0.95 bands overlap at approximately R=0.040–0.111. Matched-local BPTT gives essentially the same overlap. Requiring every measured minibatch/layer cosine >=0.95 narrows the shared span to about R=0.040–0.091. These spans connect neighboring passing samples; they do not certify all intervening values or exact threshold locations.

**Practical starting target: R≈0.075.** It sits inside both bands. This is a proposed operating target, not a demonstrated training optimum:

| Block | Injected B | Measured R | Worst pooled cosine | Worst minibatch/layer cosine |
|---|---:|---:|---:|---:|
| 1 | 0.0045817721 | 0.075523 | 0.997933 | 0.977892 |
| 2 | 0.0008414149986 | 0.075539 | 0.964795 | 0.978754 |
| 3 | 0.01704487339 | 0.075614 | 0.996350 | 0.985336 |

Interpretation: supports the bounded H-002 claim of a common useful range at this initializer. Relative RMS is a workable beta-calibration target, but does not uniquely determine alignment: the three block curves differ. Block 2 is limiting; its best minimum pooled cosine in this focused grid is 0.96735 near R=0.09085. No shared >=0.99 range was found. Its small-R degradation remains unexplained; this experiment does not identify a cause.

An absolute-RMS overlap also exists, approximately 2.64e-4–5.71e-4 for pooled >=0.95. Thus overlap alone does not establish that relative scaling is uniquely better. One checkpoint/seed and one cohort do not validate either rule across training or architectures. The same beta values may produce different displacement after weights or BN statistics change.

Decision: carry R≈0.075 and the measured blockwise B vector into the proposed 10-epoch training design, with matched smaller/larger controls chosen when that training contract is frozen. Do not launch training from a BPTT runner that ignores beta. Keep the accuracy question separate from this successful gradient-alignment diagnostic.

Evidence: `results/cifar-block-rms-focused-20260929-v1/run/`, `analysis/summary.json`, `analysis/batch_summary.json`, `analysis/common_rms_analysis.json`; exact vectors in `configs/cifar/block_rms_focused_20260929.json`.

[Relative and absolute RMS figure](../../../../../results/cifar-block-rms-focused-20260929-v1/analysis/common_rms_cosine.jpg)
