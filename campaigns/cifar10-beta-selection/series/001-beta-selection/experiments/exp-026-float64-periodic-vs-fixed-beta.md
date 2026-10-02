---
id: "exp-026"
title: "Five-epoch float64 ours: periodic RMS calibration versus epoch-five fixed beta"
status: "cancelled"
hypotheses: ["H-020"]
---
# Does controlling the response during training improve learning?

Filip explicitly authorizes two fresh five-epoch ours runs on two RTX5090s.
This new assignment supersedes the earlier training stop only for these arms.
Follow [H-020](../../../hypotheses/H-020-periodic-float64-beta-training.md) and
the selected local-comparison targets from [exp-025](../results/exp-025-all-scheme-layer-rms.md).

## Matched comparison

- **periodic:** calibrate at the start and approximately25/50/75% through
  each epoch:20 calibrations total. On1407 minibatches, run before updates
  1/353/705/1057 (completed batches0/352/704/1056). Use the same256 training
  images, calibration batch32, no augmentation, 3% RMS tolerance and at most24
  evaluations per block. Warm-start the previous beta; commit only complete
  successful vectors and preserve training state/BN/RNG. The initial search
  starts from the exp-025 initialization-calibrated values.
- **fixed_e5:** use betas
  **[0.3081665462857997,189.5629586505938,0.46590251600673055]** throughout.
  These produced output d/f **[0.01,0.005,0.00001]** on the saved ours EqProp
  epoch5 checkpoint in exp-025. This is calibration on a prior trajectory;
  achieving the same RMS at epoch5 of the new run is an outcome, not assumed.

Both restore the same seed0 epoch0 tensors, fresh Adam and scheduler. CIFAR10
45k/5k train/validation split, batch32, validation batch16, same deterministic
augmentation/order, trainable BN affine and gains, ours V4/C1, blocks[3,3,2],
free/nudged T/K=[6,6,4], no read noise or official test access. Use the original
ours_edge_head_down_e10 learning rates (head5.334108501991406e-5) and50-epoch
cosine schedule; stop after5 epochs/7035 updates without LR tuning.

All network computations, states, BN, forcing, gradients and optimizer moment
tensors use float64. Preserve the earlier input contract: preprocessing in
float32, promoted before the network. Hybrid training obtains boundary/digital
gradients through the free T pass and replaces only the eight convolution
gradients with stable centered EqProp. It remains distinct from full-network
BPTT's detached T plus tracked K pass.

Measure actual training responses from existing phase pairs, without extra
solves; retain samples at each calibration, first/last minibatch and every100
minibatches. Log beta, RMS, loss, accuracy and calibration time. Disable separate
gradient-cosine/boundary replays and scientific preflight audits. No training
smokes, following Filip's standing request; focused implementation tests,
initializer identity and nonfinite guards remain. Evaluate every validation
epoch, preserve full-state epoch checkpoints, and compare epoch5 rather than
selecting the best epoch. No automatic extension.

## Execution and monitoring handoff

Root: `results/cifar-float64-periodic-beta-20261001-v1/`.
Configs: `configs/cifar/float64_periodic_beta_20261001/`.
Cells: `periodic`, `fixed_e5`; source snapshot and prepared assets live under
the same root. Queue placements: Fifi and Loulou, one RTX5090 per arm; no
other GPU types. Respect the two-distinct5090 daytime cap and authorized Ben
sharing. Reserve28GiB peak plus3GiB headroom per arm; refine observed usage
without changing scientific settings. Exp-025 float64 replay peaked20.36GiB;
training adds Adam and retained state.

Budget12GPU-hours per arm including attempts/validation; total24GPU-hours.
The initial5–8hour estimate was too low. First99 update intervals average
10.17s/step (periodic) and10.19s/step (fixed): about4hours training per epoch,
or4–4.5hours including validation/calibration. Five epochs therefore project
to20–23hours per arm. **Budget risk: the current12-hour cap cannot complete
the assigned five epochs at this throughput.** It remains unchanged pending
a revised execution decision. Epoch-boundary pauses preserve state when the
remaining allowance cannot fit another epoch.
No overnight-only window; these explicitly requested runs may cross midnight.
Queue admission deadline October2 18:00 Paris, without extending compute caps.

Shared run-watch service owns monitoring and CPU collection; root owns review.
Jobs: `cifar-fp64-beta-periodic-20261001` and
`cifar-fp64-beta-fixed-e5-20261001`; prepared queue specs are in `launch/`.
Inspect with installed `gpu_queue.py status JOB_ID`. Queue completion invokes
`python -m experiments.run_cifar_float64_beta_pair collect ARM` from the frozen
source directory; target validation uses the same entrypoint with `validate`.
It verifies five complete45k/5k epochs,7035 updates, fresh exact initializer,
float64 model/Adam state,20 versus0 calibrations, BN updates, immutable inputs
and canonical bundle hashes, then writes comparison JSON/JPG in `analysis/`. Stop/escalate on nonfinite values, failed calibration, changed
source/initializer, invalid precision or exhausted budget. Preserve interrupted
bundles and recover only through a full-state checkpoint in a new linked bundle.

Current state: both queue jobs cancelled at Filip's request on October 1,
around14:34 Paris, before epoch1 completed. Queue receipts confirm both workers
exited and reservations were released. Collected local cells preserve all artifacts;
only initialization checkpoints exist. Periodic's raw status says failed because
SIGTERM also terminated a DataLoader worker; this was cancellation, not a
scientific/numerical failure. Fixed reports a planned pause. Neither may resume
automatically. See the [partial result](../results/exp-026-float64-periodic-vs-fixed-beta.md)
and [performance pilot](../../../../pilots/cifar-float64-performance-20261001.md).
Frozen training source identity remains
`f172e1e210b255c00b42a80027ade0a0e9e93665d22dcc0ebb6650201365c2ef`.
