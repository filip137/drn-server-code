# Does a nonlinear resistive readout widen the useful initial RMS range?

User-authorized initialization-only follow-up to the
[CIFAR beta question](../cifar10-beta-selection/README.md). Compare the existing
single dense readout with a coupled resistive head containing 512 physical
perfect-diode hidden units (256 of each polarity) and 10 linear outputs.
Ours V4/C1, existing conductance initialization/bounds and input gain 100;
no new BN, dropout or learned bias. Candidate head uses four asynchronous
iterations; convolutional T/K remains [6,6,4]. Head derivatives use autograd.

Both restore the same seed 0 convolutional/BN/gain initializer. The new head
uses an independent seed 0 RNG scope. Parameters and BN buffers stay frozen.
Use the first 128 images of the established 256-image training cohort, four
batches of 32, no augmentation/read noise/official test. No optimizer or training.

| Block | Output relative centered RMS targets |
|---|---|
| 1 | .0005, .0015, .005, .015, .05, .15 |
| 2 | .02, .03, .05, .075, .1, .15 |
| 3 | .0005, .0015, .005, .015, .05, .15, .3 |

Calibrate beta per head/block/target in float64 to 3% within 12 evaluations,
warm-start adjacent targets; replay the same beta in float32. Compare both against
the head's common float64 matched block-local BPTT; retain own-precision local
comparisons separately. Record every layer/minibatch and pooled cosine, norm,
own/output RMS, beta and forcing. Require cosine >0.95 in every layer and all
four minibatches. Compare contiguous sampled passing spans: >=3x improvement
in upper/lower RMS ratio is material, with blocks 1/2 regressions reported.
Unbracketed boundaries remain unresolved. Calibration misses are missing
measurements, not scientific cosine failures. No automatic grid expansion.
Identical trunk free states isolate the changed loss force at initialization;
this does not establish learned behaviour or improved training accuracy.

## Execution and monitoring handoff

Root: `results/cifar-nonlinear-readout-init-20261001-v1/`. Cases `single`,
`hidden512`; frozen source, inputs and queue specs under this root. Two
authorized RTX 5090 jobs, 30 GPU-minutes each including validation, total 60.
Expected 8–15min/case, subject to contention. Runner cap 1740s, queue cap 1800s;
admission expires 24h after submission. No automatic recovery/retries.
Reserve 28 GiB peak plus 3 GiB headroom per job; preserve unrelated work and
respect the weekday two-GPU fleet cap. Shared run-watch queue owns admission
and monitoring; root owns collection/review. Failed/partial attempts remain
under the same root. Stop on budget exhaustion, OOM, nonfinite or identity
failure. No training smokes or retired qualification gates.

Status: complete and reviewed. CPU implementation checks: 64 passed, two CUDA
skips; replay checks: seven passed; analyzer edge checks passed. All 19 targets
matched in each arm, producing 100 pooled and 400 layer/batch comparisons per
arm. Both remote validators and local paired analysis passed. Source/data/cohort,
trunk parameters, and all 24 free-state captures match exactly across heads.
Parameters and BN buffers remained unchanged; official test data was not read.

- Single: `cifar-nonlinear-head-single-init-20261001`, Fifi, 364s replay.
- Hidden512: `cifar-nonlinear-head-hidden512-init-20261001`, Loulou, 375s replay.

Run bundles: `{single,hidden512}/`; queue specs and terminal receipts:
`launch/`; frozen initializers/configs: `inputs/`; executed source: `source/`,
all under the result root. Both jobs exited successfully and released their
reservations. Total queue charge: 744s (12.4 GPU-minutes), within 60 minutes.
No retries or failed attempts. Peak reserved memory was 20.4/20.5 GiB. Validate with `python -m experiments.replay_cifar_nonlinear_head
--validate --run-dir BUNDLE`; regenerate paired analysis with
`python -m experiments.analyze_cifar_nonlinear_head --root RESULT_ROOT`.

## Results and interpretation

Contiguous **sampled** output relative-RMS spans with cosine >0.95 for every
layer in all four batches, against each head's common float64 local BPTT:

| Precision | Block | Single head | Hidden512 |
|---|---:|---:|---:|
| float32 | 1 | .0015–.05 | .005–.05 |
| float32 | 2 | .05–.1 | .02–.15 |
| float32 | 3 | .0015–.15 | .0015–.15 |
| float64 | 1 | .0005–.05 | .0005–.05 |
| float64 | 2 | .02–.1 | .02–.15 |
| float64 | 3 | .0005–.15 | .0005–.15 |

The primary hypothesis is unsupported at initialization: block 3's sampled
span does not widen in either precision (1x, below the declared 3x threshold).
At RMS .0005 its float32 worst-batch cosine actually falls from .901 to .809;
float64 gives .9999 for both. At .3 both heads fail in both precisions (~.91).
The extra head therefore does not remove the block 3 precision floor or the
large-displacement loss of agreement in this pilot.

There is a block 2 benefit: the float32 sampled span ratio improves 3.75x,
while block 1 narrows to 0.30x. Hidden512 passes every sampled block 2 target,
so its limits remain unbracketed. Every float64 lower limit is also a grid
edge, not an established numerical floor. This is mixed initialization
evidence, not a training-accuracy result. Do not replace the default head or
launch training from this initialization evidence alone. The following user
request separately authorizes a trained comparison.

[Comparison plot](../../results/cifar-nonlinear-readout-init-20261001-v1/analysis/cosine-vs-output-rms.jpg),
[summary](../../results/cifar-nonlinear-readout-init-20261001-v1/analysis/summary.json),
[layer/batch measurements](../../results/cifar-nonlinear-readout-init-20261001-v1/analysis/per-layer.csv).

## Authorized five-epoch BPTT follow-up

October 1: Filip requests five BPTT epochs, then the same diagnostic at epochs
1 and 5. Train both heads from the exact initializers above: float32, seed 0,
batch size 32, T/K=[6,6,4], existing Adam rates (both new head matrices use the
existing dense rate), trainable BN/gains, unchanged 50-epoch cosine schedule,
45,000 augmented training  / 5,000 validation examples. Save full epoch 1/5
checkpoints. No official test, GPU smokes, solver audits, or automatic expansion.

Replay each checkpoint in float32/64 on the same four batches; preserve local-reference
and strict all-layer/all-batch cosine >0.95 criteria. Trained weights may now
differ across architectures. Compare matching epoch trajectories separately.
Use 19 RMS targets around prior trained-network evidence (lean default):
blocks 1/2 [.001,.0025,.005,.01,.025,.05];
block 3 [.00001,.00005,.00015,.0005,.0015,.005,.015]. This shifts the original
initialization grid; do not interpret unlike sampled grids as measured boundary
movement. Calibration misses remain missing evidence, with no grid expansion.

Root: `results/cifar-nonlinear-readout-bptt-20261001-v1/`; configs/initializers
in `inputs/`, training in `train/{single,hidden512}/`, diagnostics in
`replay/e{1,5}/{single,hidden512}/`. Queue owns two RTX 5090 jobs, expected
75–110 min each, cap 3h each including two diagnostics (training cap 2h;
replays 29min each). Memory admission uses workload peak plus headroom;
no fixed 31 GiB minimum. No automatic compute retries.
The pipeline trains, replays both epochs, validates and automatically collects;
paired JPG/CSV analysis follows collection of both arms. Root owns interpretation.
Submitted jobs: `cifar-nonlinear-head-single-bptt5-20261001` and
`cifar-nonlinear-head-hidden512-bptt5-20261001`; prepared placements Fifi/Riri.
The shared queue service (PID 1124310) is active and owns admission/monitoring
at 60s cadence; collection callbacks validate both saved checkpoints and bind
each diagnostic to its actual training checkpoint. Source is frozen under
`source/`; exact job/collection commands are in `launch/{single,hidden512}.json`.
Admission expires October 2 16:03 UTC. CPU checks: 37 training/compatibility
passes (two CUDA skips), 14 replay checks and three pipeline checks.
Memory-policy update: Filip explicitly removed the fixed 31 GiB rule. The stale
experiment-loop skill requirement is corrected. Pending hidden512 now declares
22 GiB peak plus 3 GiB headroom (25 GiB total), based on its measured 20.47 GiB
float64 replay peak and the control's 10.85 GiB BPTT peak. Its never-started
queue contract was updated while held, preserving ID, budget, deadline and
all scientific settings; the original spec remains in
`launch/hidden512-before-memory-update.json` and the queue event history.
The job has been released. At the subsequent check Riri had only 7.9 GiB
free after new unrelated work arrived, so admission remains queued for
Fifi/Riri; the fixed reservation is no longer the blocker.
Scientific interpretation remains pending completion; paired analysis is
written automatically to `replay/e{1,5}/analysis/` and `analysis/comparison.json`.
No separate monitor or training continuation is armed.
