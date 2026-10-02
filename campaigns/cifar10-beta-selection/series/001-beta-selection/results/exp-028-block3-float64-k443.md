---
experiment: "exp-028"
evidence: "validated-local"
summary: "Block3-only float64 passes EqProp epochs1/5 but float32 block2 fails at initialization and BPTT epoch10; same-K all64 nudging passes all four."
verdicts: {"H-023": "contradicts"}
---
# Float64 only in block3 is checkpoint-dependent

All four256-image checkpoint replays completed and validated on Loulou RTX5090,
PyTorch2.11.0+cu128/CUDA12.8. Candidate free pass/forcing is32, nudged K=[4,4,3],
with nudging/formula32 in blocks1/2 and64 in block3. Free T=[6,6,4]. The reference
is full64 block-local BPTT at original K=[6,6,4]; it is not full-network BPTT.
All32 versus64 operations disable TF32 and use deterministic execution.

Each number below is the worst cosine across every layer and eight minibatches
in that block. All norm ratios must also remain within0.9–1.1 for a pass.

| Checkpoint/history | Block1 (32) | Block2 (32) | Block3 (64) | Every layer/minibatch passes |
|---|---:|---:|---:|---|
| Initialization | 0.997965 | **0.559414** | 0.999756 | No |
| EqProp epoch1 | 0.952302 | 0.996570 | 0.999807 | Yes |
| EqProp epoch5 | 0.996936 | 0.999680 | 0.999944 | Yes |
| BPTT epoch10 | 0.981451 | **0.827456** | 0.999983 | No |

All failures are in block2:16/24 layer/minibatch comparisons at initialization
and12/24 at BPTT10. Maximum gradient norm ratios there are1.7360 and1.2115.
Blocks1/3 pass both cosine and norm requirements at every requested checkpoint.
Epoch1 block1's0.9523 is also present in full64: finite-beta/reference mismatch
remains and is not caused by the proposed precision change.

## What the control isolates

The control uses the identical32 free states/forcing, candidate beta and K443,
but nudges all blocks in64. It passes every layer/minibatch at all four checkpoints.
Block2's worst cosine improves0.5594→0.999920 at initialization and
0.8275→0.995422 at BPTT10. Thus these failures are precision failures, not a
necessary consequence of reducing K. Original-K full64 EqProp also passes all
four. Control and candidate share the identical block3 computation, reused rather
than repeated; the additional control solves concern blocks1/2.

Despite matching output RMS, block2's smallest internal relative displacement is
approximately2.35e-8 at initialization and2.16e-8 at BPTT10, versus2.94e-6 and
2.47e-6 at EqProp1/5. This is consistent with earlier float32 response-floor
measurements: output displacement alone does not ensure resolvable inner-layer
responses. These four points do not define a universal displacement threshold.

The requested block3-only64 policy is contradicted across this checkpoint set.
It works on the two tested EqProp-trained checkpoints, but age alone cannot justify
a precision switch: BPTT10 belongs to a different trajectory. This is not evidence
that the EqProp trajectory deteriorates between5 and10.

A candidate supported by the per-block evidence is32 nudging in block1 and64 in
blocks2/3, retaining K443. Combining those selected block gradients would meet the
same read-only criteria here; its training accuracy and runtime remain unmeasured.
No training run or further beta/precision search is launched by this review.

## RMS calibration and beta

Targets are relative centered output RMS[0.01,0.005,0.00001]. Candidate precision/K
was used for calibration on all256 images. Every prior native-K beta guess already
met the3% tolerance after shortening K; one endpoint measurement per block sufficed.
Betas are unchanged from their initialization guesses, not assumed to hit the targets.
Achieved output RMS ranges across the four checkpoints are0.009756–0.009994,
0.004974–0.005000 and9.9993e-6–9.9998e-6 for blocks1–3.

| Checkpoint | Beta block1 | Beta block2 | Beta block3 |
|---|---:|---:|---:|
| Initialization | 0.0005857333 | 0.00005431466 | 0.000002204854 |
| EqProp1 | 6.671724 | 522.8088 | 1.645598 |
| EqProp5 | 0.3081665 | 189.5630 | 0.4659025 |
| BPTT10 | 0.04697655 | 0.5144223 | 0.01165072 |

## Timing and evidence

Two matched35-update timing probes completed, with5 warmups/30 measured each.
Both restore the same initializer and use the same initialization-calibrated beta,
float32 free/autograd/Adam, free T664 and nudged K443; only the nudged precision
of blocks1/2 changes. Config/source/checkpoint/environment and first-update
prediction/loss/forcing identities match exactly.

| Nudged precision | Mean s/update | Allocated peak GiB | Reserved peak GiB | Projected epoch compute min |
|---|---:|---:|---:|---:|
| All blocks64 | 2.885285 | 10.60376 | 13.78906 | 67.66 |
| Only block3 in64 | 1.041587 | 10.60400 | 11.25781 | 24.43 |

This is2.770x throughput,63.90% less update time. Mean times over each consecutive
10-update window remain2.8850–2.8856s versus1.0408–1.0423s. Reserved GPU memory
falls2.531GiB while whole-update peak live allocation is effectively unchanged.
Allocated/reserved are PyTorch allocator measurements and exclude some driver/context
memory. The queue observed only our compute process on Loulou during this assignment;
this is still one allocation sequence, not a replicated hardware benchmark.

Epoch projections use1407 updates and exclude data loading, validation and beta
recalibration; no full epoch or learning-accuracy result was measured. The speed
benefit is real in this diagnostic, but the proposed precision policy fails the
required gradient checks at two checkpoints. Do not restart training with it as a
universal policy. [Timing/memory JPG](../../../../../results/cifar-block3-float64-k443-20261001-v1/analysis/timing_memory.jpg).

All six queue jobs completed, with one attempt and successful target/local validation
each. Charged runtime776.27s (12.94GPU-min), within the4200s study cap; no failed,
superseded or pending jobs. `analysis/queue_closeout.json` records handles/accounting.

Root:`results/cifar-block3-float64-k443-20261001-v1/`, cells`init-ours-e0`,
`eqprop-ours-e1`, `eqprop-ours-e5`, `bptt-ours-e10`, `timing_all64`, `timing_block3`.
All checkpoint/source/config identities and frozen parameter/BN checks passed.
Each replay contains192 batch rows and24 pooled rows: three modes, eight convolution
weights, eight minibatches of32. Four focused CPU tests passed before launch,
including a real tiny256-image replay, precision/K routing and calibration-failure
preservation. Shared run-watch owns collection and target/local validation. Detailed
source hashes, queue receipts, beta histories, source snapshot, logs and raw tensors
remain outside Git under the result root. No official test split was accessed.

[Every-layer cosine JPG](../../../../../results/cifar-block3-float64-k443-20261001-v1/analysis/layer_cosines.jpg)
· [Summary JSON](../../../../../results/cifar-block3-float64-k443-20261001-v1/analysis/summary.json).
