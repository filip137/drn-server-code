---
id: "exp-005"
title: "Expand the RMS search to locate blockwise cosine peaks and upper tolerances"
status: "complete"
hypotheses: ["H-002", "H-003"]
---
# Expanded RMS search

Filip requests expanding the search after asking whether the best RMS depends on
block size. Extend exp-003/004 without changing their scientific computation:
same seed0 checkpoint, 256 training images, batch32, native endpoints, no noise,
T=K=[6,6,4], and each scheme's own frozen forces/native and local BPTT references.
Baseline V1/C1, ours V4/C1, legacy V4/C0.25. No training or smokes.

Relative centered output RMS targets:
`[0.0005,0.001,0.002,0.004,0.007,0.01,0.014,0.02,0.035,0.045,0.055,0.065,0.075,0.0825,0.10,0.125,0.30,0.40,0.55,0.75,1.0,1.5]`.
Calibrate each block/scheme separately within3%, with24 evaluations/target at
most; reuse the validated runner. Targets0.02 and0.075 overlap the prior sweep;
reuse prior evidence for other existing targets. Preserve all calibration history.

Decision: report the best sampled worst-layer pooled cosine and the largest
tested passing displacement at thresholds0.95/0.99, retaining disconnected bands
and per-batch variation. Distinguish true interior peaks from boundary maxima
and nearly flat plateaus. Depth remains confounded with block position/width;
this cannot isolate a causal depth effect or establish a training optimum.
Relate expanded bands to H-002/H-003; new points may bound their scope.

## Execution and monitoring handoff

Root: `results/cifar-block-rms-expanded-20260929-v1/`, with `ours/`, `legacy/`,
and `baseline/` bundles, frozen sources/inputs, job specs and CPU collectors.
Configs: `configs/cifar/block_rms_expanded_20260929/{ours,legacy,baseline}.json`.
Ours/baseline use Loulou5090 sequentially; legacy uses Trex5090. Shared queue
owns admission, monitoring and collection; respect daytime two-GPU cap and Ben
sharing. Expected6–10min/scheme; hard1200s/job,3600s combined including attempts.
Deadline October1 08:00 Paris; stop on nonfinite output, exhausted budget or
unattainable target and preserve partial evidence. No scientific recovery changes.

Jobs: `cifar-rms-expanded-{ours,legacy,baseline}-5090-20260929`.
Inspect using `python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID`.
Collectors are each `analysis/collect.py`; validate canonical bundles and22x8
pooled/22x8x8 batch rows,66 calibrated targets, unchanged model/BN and displacement
agreement between calibration and replay. All three queue jobs and local collections
are complete, with66 calibrated targets and176 pooled/1408 batch rows each.
No failures;926.4s combined charged runtime. Sources/inputs and numerical runner
identities are preserved. [Reviewed result](../results/exp-005-expanded-rms.md)
records peaks, upper tolerances and the finer-grid qualification of passing bands.
