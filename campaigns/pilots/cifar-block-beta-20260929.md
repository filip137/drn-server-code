# CIFAR blockwise beta calibration

Filip requests a5090 calibration giving output RMS displacement1 for two-conv
blocks and4 for three-conv blocks. Working defaults, stated before execution:
ours at original seed0 initialization; centered relative physical-output RMS
`RMS((s_plus-s_minus)/2) / RMS(s_free)`. Also report absolute RMS. User answers to
the pending reference-state/metric questions supersede these defaults.

CIFAR L8 has block depths[3,3,2], widths[128,256,512]; solve a separate beta for
each block, targets[4,4,1]. Preserve the native amplificationV4/C1, explicit
perfect-diode solver, T=[6,6,4], K=[6,6,4], dtypefloat32, data preprocessing and
initializer. No training or optimizer updates, no noise, no readout-beta target.
This is finite-iteration frozen-current calibration, not a solver-convergence or
EqProp-gradient qualification. The current CIFAR trainer is BPTT and does not
consume beta; selected values apply to the declared diagnostic nudging contract.

Compute the loss signal by differentiating summed classification CE through the
native free forward pass, including digital pooling/BN and downstream blocks.
At each physical block output freeze that derivative. Hold the block input and
all parameters fixed; start each +/-K-step branch from the identical saved
post-T block state. Add `+/- B*dCE_sum/ds` to the block output's linear coefficient.
Report this injected coefficientB explicitly, avoiding ambiguous amplification
rescaling. BN uses batch32 minibatch statistics with running buffers frozen.
Use256 deterministic training examples:128 calibration and128 disjoint confirmation,
no augmentation or official-test data. Pool squared norms across examples/nodes,
then take their ratio. Target tolerance5% on calibration; report confirmation
error independently (<=10% is an exploratory transfer check, not a retuning gate).
Search bounded positive beta on a logarithmic bracket; max24 evaluations/block,
fail visibly if targets cannot be bracketed, values become nonfinite or budget ends.
Verify all model/BN values unchanged before/after and preserve measurement histories.

Result root registered before creation:
`results/cifar-block-beta-20260929-v1/`.
Use one shared-queue default worker on an eligible lab RTX5090, respecting weekday
fleet quota and existing users. Expected5–15min, cap1 GPU-hour, no training smokes.
Report per-block beta, free RMS, absolute/relative displacement and confirmation
error here after local validation. No subsequent training is authorized.
Queue job `cifar-block-beta-init-5090-20260929` submitted for Loulou RTX5090; shared queue owns placement, monitoring, bounded operational recovery and collection. Config `configs/cifar/block_beta_20260929.json`; frozen source identity and launch spec under `results/cifar-block-beta-20260929-v1/{source/source_identity.json,launch/job.json}`. Two focused CPU numerical tests passed: injected-current coefficient/zero-beta symmetry and nonlinear beta search. No training smoke. Routine observations require `gpu_queue.py status cifar-block-beta-init-5090-20260929`; completion hook validates and writes this note’s results.

## Results

All three calibration targets matched within5%; model and BN values were unchanged.

| Block | Conv layers | Target relative RMS | Injected B | Calibration relative RMS | Confirmation relative RMS |
|---|---:|---:|---:|---:|---:|
| 1 | 3 | 4 | 0.34178884 | 4.015 | 4.2586 |
| 2 | 3 | 4 | 0.063142913 | 4.0195 | 4.2334 |
| 3 | 2 | 1 | 0.32483228 | 1.0463 | 1.0916 |

B is the signed linear-coefficient multiplier of the frozen summed-CE derivative at each physical block output. It is not an implicitly amplification-rescaled training beta.
These values describe native finite-T/K centered nudging at initialization. Confirmation errors are recorded separately; they do not trigger hidden retuning. No training or gradient-fidelity claim follows from matching displacement.

Evidence: `results/cifar-block-beta-20260929-v1/analysis/summary.json` and the canonical `run/` bundle.

Reviewed locally: all three calibration targets meet5%, and all three disjoint confirmation errors meet10% (6.46%,5.84%,9.16%). Calibration compute took40.5s on Loulou5090. Values are conditional on initialization, batch32 BN statistics, summed-CE force and native finiteT/K; they are not guaranteed at later checkpoints or under a different loss/current normalization. Each three-conv block needs its own B despite having the same target.

## Assigned gradient comparison

September 29: compare centered EqProp at the three selected B values with BPTT.
Reuse the exact initializer, 256 indices, batch32, no augmentation/noise, and
T/K=[6,6,4]. No parameter updates or BN running-buffer changes. Measure every
Conv weight tensor; beta and iterations remain fixed. Primary practical reference:
the native CIFAR trainer's T-free/K-tracked full-network CE gradient. Also report
a matched local reference: K unnudged tracked steps starting from the saved
post-T block state, with its input and calibrated output loss derivative frozen.
The latter isolates local estimator alignment; it is not full-network training.
Use the repository's centered EqProp parameter-gradient formula. Average gradients
over examples before measuring cosine, and retain per-batch cosine and gradient
norms to expose dead/nonfinite references and batch variation. Report alignment
descriptively; displacement matching does not establish gradient fidelity.

Result root: `results/cifar-block-beta-gradients-20260929-v1/`.
One Loulou RTX5090 queue worker, expected2–5min, total cap1 GPU-hour. The shared
queue owns placement, monitoring and collection; no training smokes. Completion
requires all eight Conv layers, both references, unchanged weights/BN, and local
bundle validation. Preserve failed attempts; no further training is authorized.

Queue job: `cifar-block-beta-gradients-5090-20260929`. Config: `configs/cifar/block_beta_gradients_20260929.json`; frozen source/launch/collection artifacts in the gradient result root. CPU numerical test verifies centered-gradient sign and batch-mean normalization against BPTT on a single-layer network (3 calibration/comparison tests pass). Uses the original calibration source snapshot plus the new comparison runner.

Operational replacement: `results/cifar-block-beta-gradients-20260929-v2/`, queue `cifar-block-beta-gradients-5090-20260929-v2`. The original v1 attempt failed in the native ConvResistive energy derivative (72 GiB broadcast temporary); no measurements completed. The diagnostic now evaluates the same squared-difference derivative from squared norms and cross-products, preserving all amplification factors, batch averaging, beta, phases, and float32 solver settings. CPU tests match the original formula at V=1 and V=4 and verify gradient sign/batch normalization (5 tests pass). Frozen failed source/evidence retained; replacement allowance3591s preserves the original3600s total after the8.45s failed attempt.

Numerical correction: v2 completed256 examples (68.5s), but subtracting separately reduced float32 energies suffers cancellation of the large fixed-input squared term; early-layer cosine/norm results are superseded. `results/cifar-block-beta-gradients-20260929-v3/`, queue `cifar-block-beta-gradients-5090-20260929-v3`, computes the identical centered formula using `(d_plus-d_minus)*(d_plus+d_minus)` before reduction. The zero boundary-difference term cancels exactly. Added CPU comparison to the original double-precision formula, including large boundary voltages and amplified internal edges; all6 tests pass. Repeats unchanged phases, forces, B, T/K and cohort within the original budget. v2 remains locally preserved.

## Gradient comparison results and review

Final v3: all256 images /8 batches /8 Conv weights completed; both references finite and non-dead, model weights and BN buffers unchanged. Native finiteT/K, original initialization, no read noise. Compute55.1s on Loulou5090; peak reserved10.21GiB. CPU collection validates the canonical bundle. v1 failed/OOM and v2 numerically unstable subtraction remain preserved; only v3 is used below.

Cosine of the gradients averaged over all256 examples, with native-BPTT batch variation shown separately:

| Block / Conv | B | Native BPTT cosine | Matched local BPTT cosine | Batch cosine min / median / max | EP / native norm |
|---|---:|---:|---:|---|---:|
| 1 / 1 | 0.34178884 | -0.1357 | -0.1419 | 0.0688 / 0.3315 / 0.8925 | 0.1926 |
| 1 / 2 | 0.34178884 | 0.6394 | 0.6375 | 0.5408 / 0.5867 / 0.7154 | 0.5931 |
| 1 / 3 | 0.34178884 | 0.0366 | 0.0369 | 0.0186 / 0.0259 / 0.0894 | 6.0761 |
| 2 / 1 | 0.063142913 | -0.0461 | -0.1084 | -0.2936 / 0.4229 / 0.9316 | 0.4473 |
| 2 / 2 | 0.063142913 | 0.6259 | 0.6178 | 0.4845 / 0.5806 / 0.6864 | 0.6143 |
| 2 / 3 | 0.063142913 | 0.1393 | 0.1393 | 0.1063 / 0.1500 / 0.2452 | 3.4058 |
| 3 / 1 | 0.32483228 | 0.7041 | 0.7029 | 0.5827 / 0.7967 / 0.8837 | 0.5628 |
| 3 / 2 | 0.32483228 | 0.8880 | 0.8881 | 0.8352 / 0.8609 / 0.8747 | 0.7842 |

Review: matching relative output displacement does not provide close BPTT gradient directions for this CIFAR architecture at the native finite iteration counts. The target4 blocks are particularly weak: output-Conv cosine0.0366/0.1393 and gradient norms6.08x/3.41x native BPTT. Target1 block has stronger alignment0.7041/0.8880, but still imperfect. Matched-local references lead to the same broad conclusion, so the frozen-force versus native-loss difference alone does not explain it. First-Conv pooled cosines are negative, but minibatch medians are positive; do not describe every batch as anti-aligned.

This does not isolate the cause between large nudging/nonlinear response and insufficient free/nudged settling. A next discriminating test would lower B at fixed T/K; only if alignment remains poor would a longer-iteration comparison separate that limitation. These are proposals, not queued follow-up runs. No training accuracy conclusion or beta recommendation follows from displacement matching alone.

Evidence: `results/cifar-block-beta-gradients-20260929-v3/analysis/summary.json`, `results/cifar-block-beta-gradients-20260929-v3/analysis/batch_summary.json`, and `results/cifar-block-beta-gradients-20260929-v3/run/artifacts/comparison.json`. Queue `cifar-block-beta-gradients-5090-20260929-v3`; source identity and exact config are retained in the bundle. Six focused CPU numerical tests pass, including stable centered subtraction against a double-precision reference.

## Adopted campaign

Follow-up beta selection now belongs to [CIFAR-10 beta selection](../cifar10-beta-selection/README.md). The authorized smaller-beta sweep is [exp-001](../cifar10-beta-selection/series/001-beta-selection/experiments/exp-001-smaller-beta-cosine.md); the campaign also plans 10-epoch comparisons. This pilot preserves the original calibration/gradient evidence and prior result paths.
