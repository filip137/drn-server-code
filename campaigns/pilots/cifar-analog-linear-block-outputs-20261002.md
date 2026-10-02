# Analog L8 with linear block outputs: legacy and ours BPTT

## Question and authorized cases

After the [digital pilot](cifar-linear-block-outputs-20261002.md) reached 91.36%
validation accuracy, Filip requested the same output-layer change in analog L8
and two queued 10-epoch runs, legacy and ours, on RTX 5090s or RTX 3090s.
He explicitly selected **BPTT baseline** training. Zero weight decay carries
forward. This is a new pilot in the [beta-selection campaign](../cifar10-beta-selection/README.md).

Replace only the last perfect-diode activation in each block with a linear
activation on both physical halves. Keep eight convolutions, depths [3,3,2],
logical widths 128/256/512, the five internal perfect-diode layers, differential
decoding, max-pooling, affine BN, learned boundary gains and analog dense readout.
The general solver remains perfect-diode quadratic coordinate descent, with
each individual layer applying its own declared activation. No output rescaling
or added weights. Existing configs retain their nonlinear outputs.

Fresh seed-0 initializations with equal tensors across arms; float32; GPU-resident
finite-iteration BPTT with T/K=[6,6,4]. Legacy uses V4/C0.25 and ours V4/C1.
Use their established Adam learning-rate vectors and explicit diode dictionaries
from the historical floor configs; remove continuation metadata and retain only
their recipes. Trainable BN/gains/head, zero weight decay, batch 32/evaluation 16,
10 epochs (14,070 updates), first 10 epochs of the established 50-epoch cosine
schedule to a 0.02 LR ratio. No noise, EqProp, calibration or solver audits.

Training-only CIFAR10: stratified 45k/5k split with seed 0, unchanged deterministic
crop/flip augmentation and epoch order. Never open the official test files.
Primary comparison: epoch 10 validation accuracy and CE; also per-epoch curves,
best validation checkpoint, runtime and peak memory. Best checkpoint: maximum
accuracy, minimum loss, earliest exact tie. Preserve epochs 1/5/10 as well.
Compare recipes descriptively: amplification and inherited learning rates differ.
Without a newly matched nonlinear-output control, do not attribute any difference
from historical results solely to removing the output diodes. Noise effects remain
untested by this BPTT baseline.

Configs: [legacy](../../configs/cifar/linear_outputs_bptt_20261002/legacy.json)
and [ours](../../configs/cifar/linear_outputs_bptt_20261002/ours.json).

## Budget, gates and outputs

Verified RTX 3090/5090 placements in the submitted specs: local, trex, fifi,
and riri. Queue checks live ownership, memory plus headroom, and the weekday
two-5090 admission cap. Preserve unrelated work; only share with Ben.
Initial allowance: **6 GPU-hours per arm, 12 total including smoke/retries**;
Trainer cap 5.9 h; queue allowance 21,590 seconds per arm after charging ten
seconds for its smoke. The short smoke suggests approximately five hours on a
3090, including startup overhead; production epoch timing will refine this.
No tuning,
extra cases or automatic epoch extension. Numerical failures retain evidence;
operational recovery must preserve science and fit the remaining allowance.

Run targeted model/restore tests, CPU initialization preparation and a real-data
GPU smoke of each exact config (four updates plus one validation batch). Keep
configured iterations without retired qualification probes. Require finite
nonzero first-batch gradients and valid bundles. Initializer tensors and split
must match across arms; separate prepared assets encode the different solvers.

All outputs and attempts:
[results/cifar10-l8-linear-outputs-bptt-20261002-v1/](../../results/cifar10-l8-linear-outputs-bptt-20261002-v1/).
Frozen source/config identity, prepared assets, smokes, queue specs/receipts,
arm bundles and CPU collection summaries live below this root. Large files stay
outside Git. Per-arm completion hooks collect remote bundles, validate all 10
epochs/14,070 steps and finite checkpoints, and generate matplotlib JPG curves.

## Execution and monitoring handoff

State: **both arms running**, verified on 2026-10-02 at approximately 09:35 UTC.
The shared run-watch queue is the sole routine monitoring owner: persistent
60-second polling, 45-minute stall threshold, bounded operational recovery and
automatic CPU collection. Both queue receipts confirm live workers and held
reservations; direct trainer status files independently show advancing updates.

| Arm | Queue ID | Placement | Trainer PID | Last directly verified progress |
| --- | --- | --- | --- | --- |
| legacy | `cifar-linear-bptt-legacy-20261002` | local / nom-cool-2, RTX 3090 | 3008187 | epoch 1, batch 300/1407 |
| ours | `cifar-linear-bptt-ours-20261002` | fifi, RTX 5090 | 2018830 | epoch 1, batch 500/1407 |

Attempt tokens: legacy `12501ee2c0ff4abc92fd23a2dbb2002b`; ours
`e7ddc3a9bceb4e94986217df4caa7832`. On each target, worker receipts and logs are
under `/home/filip/.local/state/run-watch/queue-workers/attempts/<token>/`.
Specs and saved queue observations are `launch/{legacy,ours}-job.json` and
`launch/{legacy,ours}-queue-observation.json` below the result root. Refresh with
`python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status <queue-id>`.

The local legacy bundle is `cells/legacy/` below the result root. The live ours
bundle is on fifi at
`/home/filip/server_code/results/cifar10-l8-linear-outputs-bptt-20261002-v1/cells/ours/`;
the completion hook collects it into local `cells/ours/`. Each has `status.json`,
`metrics.jsonl`, manifests and checkpoints. Collection writes
`analysis/{arm}-collection.json`, comparison data and JPG curves; scientific
review remains pending after collection.

Implementation and execution checks passed: 23 focused tests; two CUDA-only
tests skipped in the sandbox, with the actual CUDA path exercised by both
real-data smokes. The two preparations have identical splits and all 28 initial
parameter tensors match. Four-update RTX 3090 smokes passed for both exact
configs, including finite nonzero gradients in all 19 optimizer groups and
canonical bundle validation. Wall times were 9.289 s (legacy) and 9.137 s (ours);
both measured 10,260 MiB peak process memory. Admission reserves an 11,264 MiB
bound plus 2,048 MiB headroom. The tiny smoke validation CE was 4.2782 for legacy
and 48.8355 for ours; this is a startup observation, not a full validation result,
and the declared learning rates were retained.

Both workers use frozen source/configs from `source/`, with the actual dirty-tree
hashes in `source/source_identity.json`, and PyTorch 2.11.0+cu128 /
torchvision 0.26.0+cu128. Source, prepared assets and training-data hashes passed
verification for trex, fifi and riri. Loulou's verification timed out and
nom-cool-1's two staging transfers were too slow; neither was admitted. Only
our identified transfer processes were stopped. Partial staging directories
remain under the same study root on those hosts; no training attempts occurred
there and no unrelated jobs were changed.

## Results and review

Pending. Queue acceptance is not scientific completion. The reviewed digital
pilot remains unchanged; this note owns the new analog cases and their outcomes.
