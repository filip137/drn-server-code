# CIFAR float64 training performance

Filip stopped [exp-026](../cifar10-beta-selection/series/001-beta-selection/experiments/exp-026-float64-periodic-vs-fixed-beta.md)
to diagnose its roughly10.2s/update cost before further training.
**Both training jobs remain cancelled; no training resumed.**

We profiled real batch32 updates on Trex5090 with full float64, the same ours
initializer/data order/Adam and T/K=[6,6,4]. The two optimizations now used by
float64 hybrid training are:

- Skip convolution autograd gradients that EqProp immediately replaces, retaining
  the backward pass for boundary forces, BN, gains and readout.
- Reuse each original quadratic-coefficient calculation within an update.
  Separate free/nudged dispatch and clear the cache before weights change.

Float32 behavior is preserved. Solver iterations, betas and learning rates are
unchanged. Calibration was not the bottleneck: the stopped periodic arm's first
calibration took61.05s, while the fixed arm had the same training-step cost.

| Phase, seconds/update | Original | Both fixes |
|---|---:|---:|
| Free forward | 2.33 | 1.82 |
| Autograd backward | 3.29 | 1.19 |
| Two nudged solves per block | 4.66 | 3.73 |
| EqProp gradient formula | 0.36 | 0.37 |
| Total wall time | 10.66 | 7.13 |
| Peak reserved GPU memory | 21.75GiB | 8.80GiB |

The final paired check used the small periodic initialization betas and compared
109 tensors after each of two updates, including updated float64 weights:
**all218 comparisons were bit-exact** (logits/loss, boundary forces, gradients,
BN, parameters and Adam). The final source passes179 CPU tests. Earlier three
measured-update profiles establish the phase ordering; final timings above are
one measured update after warm-up per arm, not measured full-epoch times.

This is about33% less time per update, or1.49x throughput. At1407 updates,
7.13s projects to2.8hours of training per epoch before validation/calibration.
About78% of the remaining time is free/nudged solver work. The fixes help but
leave full float64 training expensive; no five-epoch relaunch is scheduled.

Evidence root: `results/cifar-float64-performance-20261001-v1/`.
[Phase plot](../../results/cifar-float64-performance-20261001-v1/analysis/phase_breakdown.jpg),
`analysis/summary.json`, per-case canonical bundles and frozen sources retain
all measurements. `validation_reference` versus `validation_cache_v2` is the
accepted pair. An algebraic sum candidate passed initialization but failed the
strict second-update small-beta comparison; it was removed from active code.
The first cache attempt was cancelled/superseded after CPU testing identified
the need to separate backend dispatch. Both excluded attempts are preserved.

Runner: `experiments/profile_cifar_hybrid_step.py`. Queue IDs share prefix
`cifar-fp64-profile-` and date20261001; all receipts/statuses are in
`analysis/queue_closeout.json`. The shared queue owned monitoring; root collected
and reviewed all outputs. Successful bundles pass validation. One rejected
candidate remains labelled failed/attention and the superseded cache cancelled;
the cancelled prototype retains a stale raw running status, overridden by its
verified queue exit receipt. Neither has a live worker or automatic recovery. Total charged profiling time
612.93s (10.22GPU-minutes), within the30-minute cap. Trex shared with Ben as
authorized; other lab jobs were untouched. Controller/host process release is
verified. Completed optimizations are source changes only; diagnostic updates
are discarded and are not training-accuracy evidence.

## Float32 and selective-float64 follow-up

**Completed: float64 nudged solves preserve the tested gradients while the
free pass, autograd, digital parameters and Adam remain float32.** This is a
diagnostic implementation; production precision is unchanged and training
remains stopped. New root: `results/cifar-split-precision-20261001-v1/`.

Trex RTX 5090, batch32, same initializer/data order/T/K/betas/Adam: four updates
per mode, first discarded as warm-up. Original float32 free graph in both
float32 and split; optimized full64 from the earlier work above.

| Precision | Median seconds/update | Projected train-only epoch |
|---|---:|---:|
| Float32 throughout | 1.79 | 42 min |
| Float32 free/autograd, float64 nudged solves/formula | 4.17 | 98 min |
| Optimized float64 throughout | 9.22 | 216 min |

These are three-update medians on a GPU shared with Ben, not measured epochs
or stable hardware speed ratios. Split timing includes promotion and donor
weight copies. The earlier 7.13s full64 profile had different contention.

Read-only replay used the same256 training images at ours initialization and
saved EqProp epochs1/5, checkpoint-specific betas for output relative RMS
targets[.01,.005,1e-5], T/K=[6,6,4], no augmentation/noise/official test and
frozen BN running buffers. Each row below is the worst of64 layer/minibatch
comparisons against the **same full64 block-local BPTT** reference.

| Checkpoint | Full32 cosine | Split cosine | Full64 cosine |
|---|---:|---:|---:|
| Initialization | 0.169896 | 0.999867 | 0.999956 |
| EqProp epoch1 | 0.808580 | 0.952304 | 0.952304 |
| EqProp epoch5 | 0.698494 | 0.996936 | 0.996936 |

Split passes cosine>0.95 everywhere and the proposed engineering screen
against full64 EqProp: minimum cosine0.999851 and norm ratios0.991187–1.006936
over all192 comparisons. Full32 fails both screens. The epoch1 local cosine
near0.952 already occurs in full64; this remaining finite-beta/reference error
is not introduced by splitting precision.

At epoch5, block3's output relative displacement is1.00e-5, but its first
layer's is1.18e-8. This illustrates why acceptable output RMS can conceal a
response that competes with float32 rounding internally. Solving both nudged
phases in64 preserves this response; casting already-computed32 endpoints
cannot restore it. These results cover ours0/1/5 only, not other schemes,
later checkpoints, long training or read noise.

[Gradient plot](../../results/cifar-split-precision-20261001-v1/analysis/gradient_quality.jpg)
and [timing plot](../../results/cifar-split-precision-20261001-v1/analysis/timing.jpg);
`analysis/summary.json`, `timing_*`, `replay_*_v2`, frozen sources and `launch/`
retain the evidence and exact betas. Runner/helper:
`experiments/compare_cifar_split_precision.py`; timing uses the profiler above.
Five CPU tests and all six completed canonical bundles pass validation.

Queue IDs start`cifar-precision-` and end`20261001`; the queue owned monitoring,
root collected/reviewed. All workers/reservations are released, with receipts
in `analysis/queue_closeout.json`. Total902.89s (15.05 GPU-minutes) within the
30-minute cap, including epoch1 added after a remaining-budget check. Two
initial replays failed before numerical work because Git metadata was queried
inside a source snapshot; recorded file hashes replace that lookup in v2.
Failed startup receipts are preserved, recovery is disabled, and no training
was relaunched.

## V100 mixed-precision timing

**Completed: 2.7873s/update projects to65.36 minutes of training per epoch**
for ours, batch32,1,407 updates. One V100-SXM2-32GB ran105 real optimizer
updates; the first five were warm-up. The remaining100 were stable: five
20-update windows project65.33–65.39minutes. Peak reserved memory13.79GiB.
This estimates update compute, excluding data loading, validation and beta
recalibration; it is not a measured full epoch or training-accuracy result.

The same initializer, initial beta vector, Adam, augmentation and T/K=[6,6,4]
were used as in the5090 split profile. Free/autograd/digital/Adam remain32;
nudged solves and the EqProp formula run64. Mean phase costs: free0.510s,
autograd0.577s, nudged solves1.534s, gradient formula0.148s. The65-minute
estimate is below the earlier shared5090 three-update estimate of98minutes,
but different contention and PyTorch versions prevent a controlled hardware
speed-ratio claim. V100 runtime: PyTorch2.5.0, CUDA12.2.

Evidence root: `results/cifar-v100-split-timing-20261001-v1/`.
[Timing plot](../../results/cifar-v100-split-timing-20261001-v1/analysis/update_timing.jpg),
`analysis/summary.json`, `timing_split/` and frozen `source/inputs/launch`
retain all evidence. Reproduce with
`MPLCONFIGDIR=/tmp/cifar-v100-mpl /home/filip/miniconda3/envs/py312/bin/python results/cifar-v100-split-timing-20261001-v1/analysis/analyze.py`.

Slurm456173 completed with exit0:0,343s charged within900s budget:
`fmu@v100`, `gpu_p13`, `qos_gpu-dev`, `v100-32g`, one GPU on`r7i1n3`.
The original t3 job456099 was cancelled while pending, zero runtime. Frozen
source tests passed5/5; six monitoring-adapter tests passed. Staged hashes,
shell syntax, scheduler contract, stdout/stderr and the collected canonical
bundle/105-update coverage passed. Receipts: `analysis/slurm-verification.json`,
`slurm-closeout.txt`, `closeout.json`. The detached observer exited after its
first poll; root retained active ownership through collection/review, with
no retry or dispatched recovery agent. Its stop marker is set. No assigned
job remains live, and five-epoch training remains stopped.
