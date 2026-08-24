# IBM OM HWA followed by pulse-resolved deployment

- Status: cap-128 device artifact and all four training arms are complete;
  the 16 predeclared held-out validation arms are pending
- Device source: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Evidence class: hardware-derived fitted preset, not raw measured traces
- Study: [`mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json`](../studies/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json)

## Question

Can off-chip BPTT learn weights that survive an IBM optimized-material (OM)
ReRAM program-and-verify deployment, or does a remaining matched deployment
gap justify a later on-chip SET/RESET recovery experiment?

The pilot deliberately stops before Tiki-Taka, LoRA, or soft SET/RESET
fine-tuning. A later recovery study is justified only if the predeclared
HWA-only arms underperform after the same physical deployment and a matched
recovery arm closes that gap.

## Device and pulse interpretation

The OM preset is IBM's optimized-material 14 nm ReRAM array fit. Its nominal
state is `w in [-1,1]` and its nominal minimum update is `dw_min=0.0949`, or
`0.04745` of the normalized full span after

```text
x = (w + 1) / 2.
```

Thus the nominal full-span arithmetic is about 21 minimum pulses. That is not
a programming-time guarantee. Sampled cell bounds, nonlinear soft-bound
response, cycle-to-cycle variation, verify reversals, and corrupt/stuck cells
change both success and cost. The HfO2 preset has a much larger nominal
minimum step (`dw_min=0.4622`, about 4.3 nominal full-span updates), which is
why HfO2 target pulse counts are much smaller. HfO2 remains a characterization
stress control and is not part of this first network pilot.

The program-and-verify controller starts directly from each sampled cell's
lower persistent bound, representing a fully RESET device after boundary
conditioning. Conditioning pulses are outside the 128 target-programming
pulse budget. Starting from logical zero would mix an arbitrary simulator
initial state into the programming result; starting from RESET instead gives
the physical procedure a declared and reproducible initial boundary.

At each verify read, the controller observes the apparent state:

```text
accept if abs(x_apparent - x_target) <= tau
SET if x_apparent < x_target - tau
RESET if x_apparent > x_target + tau.
```

Here `tau = 0.5 * dw_min / 2 = 0.023725` in normalized OM units. The adaptive
controller predicts how many identical pulses to batch, applies them
sequentially, verifies again, and reverses polarity only after the apparent
state crosses the acceptance window. It does not RESET after every rejected
SET. A trajectory stops on acceptance, a non-finite verify value, or 128
target-programming pulses.

The **apparent endpoint residual** is what the verify circuit admits and what
the DRN forward pass receives. The **persistent endpoint residual** is the
underlying programmed state after apparent write variation is removed. An
accepted residual is therefore effective endpoint programming error, not
pure write noise: it contains the acceptance window, pulse resolution,
stochastic pulse response, sampled bounds, and controller behavior. The
Wan-2022 comparator is an endpoint programming-noise model and does not expose
this pulse trajectory or the same failure/cost decomposition.

## Fixed physical identities

Each named DRN conductance tensor receives a fixed array-shaped AIHWKit OM
assignment. A cell keeps its sampled lower and upper bounds, SET and RESET
step parameters, reference, and corrupt/stuck status for the whole run.
Assignment fingerprints and all tensors are included in exact resume state.
Because the CUDA training interpreter does not import AIHWKit, each named
training, selection, or validation population is sampled once by the pinned
AIHWKit interpreter and transferred through a strict pickle-free NPZ plus JSON
receipt. The runtime verifies the request, tensor schema, seeds, source hashes,
artifact hash, AIHWKit version, binding geometry, and population fingerprint
before using the population.

The two IBM-aware training arms share the literal published-corruption
assignment sampled with seed `83001`:

- `om-published` retains every sampled corrupt/stuck cell;
- `om-repaired` keeps every healthy cell byte-identical and replaces only the
  sampled corrupt sites with independently seeded healthy donor cells.

This construction makes corruption the only physical-identity difference
between those two arms. It is a counterfactual repair control, not IBM's
hardware defect-management procedure and not device reassignment learned
from network performance.

## Off-chip HWA semantics

The optimizer owns a clean FP32 master tensor. For each minibatch, the compact
modifier:

1. maps each clean DRN conductance from `[0.1020408198, 1]` into `x in [0,1]`;
2. reuses that cell's fixed bounds and corrupt/stuck identity;
3. samples fresh accepted, failed, or corrupt terminal endpoints from the
   held-out-validated cap-128 endpoint artifact;
4. exposes the apparent endpoint for the complete forward/backward pass; and
5. restores the clean master before the ideal optimizer step.

There is no persistent physical state between HWA minibatches. This is
off-chip hardware-aware training against repeated hypothetical deployments,
not simulated on-chip incremental training. The endpoint artifact is used
only if its declared held-out adequacy gates pass. A failed gate blocks the
pilot; it does not trigger a Gaussian fallback.

The compact sampler uses deterministic reachability classes from the fixed
cell bounds. A corrupt cell keeps its sampled stuck persistent state while
fresh apparent write variation may be drawn. Non-corrupt terminal endpoint
and acceptance randomness advances once per minibatch.

## Common physical selection and held-out test

All four training arms use the same pulse-resolved full-corrupt OM selection
assignment and endpoint seed `83003`. Selection therefore asks which clean
master checkpoint is best after the same physical deployment, rather than
choosing by clean validation and testing physical behavior afterward.

The four training arms are:

1. clean BPTT;
2. 3% per-output-channel Gaussian HWA;
3. compact OM HWA with corrupt sites repaired; and
4. compact OM HWA with published corruption retained.

The selected checkpoints are evaluated on four held-out full-corrupt
assignments `83101..83104`, with endpoint seeds `83201..83204`. The primary
network comparison is the assignment-level mean, while each assignment-level
result remains visible.

Every pulse-resolved context records programming success, budget exhaustion,
SET, RESET, and total pulses, verify reads, reversals, clipping, and corruption.
It also writes `ibm_om_deployment.pt`, containing the exact apparent and
persistent endpoints, sampled population tensors, pulse counts, RNG
continuation state, binding layout, population fingerprint, device-model
digest, and source checkpoint digest. A later fine-tuning study must consume
this persistent bundle instead of redrawing or reprogramming the base.

## Device-model prerequisite

First run the two OM-only `hwa_production_cap128` characterization arms from
[`ibm-reram-om-hwa-device-model-20260822-v1.json`](../studies/ibm-reram-om-hwa-device-model-20260822-v1.json).
The focused profile keeps all 4,096 identity/repeat trajectories for the
adaptive controller at every target, but runs the one-pulse calibration only
on the 816 calibration trajectories per target. It uses lower-from-RESET only
and reads each identity's fixed sampled upper bound for analysis without
exposing that bound to the controller. Each arm therefore has 201,392
trajectories (402,784 total), rather than 671,744 per arm and 2,686,976 total
for the broader two-start, two-controller, four-preset cap-128 study. This is a
6.67x reduction in trajectories for the HWA prerequisite; the broader HfO2
and upper-start comparison remains a separately declared deferred control.

Prepare the focused characterization study with:

```bash
python -m ebl study prepare \
  --plan studies/ibm-reram-om-hwa-device-model-20260822-v1.json \
  --results-root results
```

Then launch the two declared arms with the pinned AIHWKit sampler available:

```bash
export EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python
python -m ebl characterize \
  --config examples/reram_program_verify/hwa_production_cap128_om_continuous.json \
  --output-dir results/ibm-reram-om-hwa-device-model-20260822-v1/runs/om-continuous
python -m ebl characterize \
  --config examples/reram_program_verify/hwa_production_cap128_om_corrupt.json \
  --output-dir results/ibm-reram-om-hwa-device-model-20260822-v1/runs/om-published
```

The persistent two-arm launcher is the equivalent parallel entry point:

```bash
python -m experiments.reram_program_verify.local_short_launcher \
  --study-profile hwa-om-cap128 \
  --aihwkit-python /home/filip/miniconda3/envs/aihwkit/bin/python
```

After both adaptive lower-from-RESET conditions pass their adequacy gates,
combine their endpoint and controller-calibration artifacts without refitting:

```bash
python -m experiments.reram_program_verify.hwa_model \
  --continuous-endpoint <om-continuous>/artifacts/bounded_uniform_model.json \
  --continuous-estimators <om-continuous>/artifacts/step_estimators.json \
  --published-endpoint <om-published>/artifacts/bounded_uniform_model.json \
  --published-estimators <om-published>/artifacts/step_estimators.json \
  --output data/ibm_reram_om_pv128_hwa_v1.json
```

The builder verifies OM, cap 128, continuous versus published corruption,
adaptive lower-from-RESET, half-step tolerance, calibration-only estimators,
and the endpoint adequacy result. It embeds source hashes in the output.

## Running the pilot

Prepare the tracked study before launching any native run:

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json \
  --results-root results
```

Each training run receives the exact bounded DRN as
`--teacher-weights` and the calibrated bundle as `--device-model`, for example:

```bash
export EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python
python -m ebl train \
  --config examples/mnist_relu_drn/ibm_om_hwa_pilot/om_repaired.json \
  --output-dir results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/runs/train-om-repaired \
  --teacher-weights results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237/20260820T140222.931115Z-809aec60-8da7bc60/checkpoints/weights.pt \
  --device-model data/ibm_reram_om_pv128_hwa_v1.json
```

The clean and Gaussian arms still receive `--device-model` because their
common selection metric is pulse-resolved OM deployment. Held-out validation
uses `python -m ebl validate` with one declared held-out config, the selected
checkpoint, the same bounded-DRN teacher, and the same device-model bundle.

## Training-phase evidence

The four formal training arms completed on 2026-08-23. Each arm performed 10
epochs and exactly 34,380 optimizer updates. The artifact-verified study state
is intentionally `incomplete`: all four training arms have one valid complete
run and all 16 held-out validation arms remain untouched.

| Training arm | Selected epoch | Common-assignment KL | Student accuracy | Programming success | Pulses, mean / median | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | initial (-1) | 0.17078 | 11.24% | 56.07% | 68.05 / 63 | 10.2 min |
| Gaussian 3% | 8 | 0.16697 | 11.88% | 56.07% | 68.04 / 63 | 10.2 min |
| OM, repaired training population | 9 | 0.10565 | 15.62% | 56.81% | 67.31 / 59 | 62.7 min |
| OM, published corruption | 9 | 0.10538 | 12.86% | 56.86% | 67.11 / 58 | 62.7 min |

These are selection results on the single predeclared full-corrupt assignment
`83003`, not held-out test results. Relative to clean BPTT, Gaussian HWA
reduced selection KL by 2.23%, while repaired and published OM HWA reduced it
by 38.13% and 38.29%, respectively. The low absolute physical accuracies and
the reversal of the repaired-versus-published ordering between KL and accuracy
make the four held-out assignments necessary before interpreting robustness or
claiming a need for on-chip recovery.

All four selection-population NPZ files have SHA-256
`945cc8e62266c097d847982470d4869db9778b5b68240e5e16a8602c0ca93010`
and population fingerprint
`879f08d1474e6e9d2c1cf0f96ff22d1753b77b1968225ca4bd8e951fd50cf55e`.
For training assignment `83001`, the repaired and published populations have
byte-identical healthy-cell tensors and the same 21,466-site published-corrupt
mask; the repaired population activates zero corrupt cells while the published
population retains all 21,466. Every selected checkpoint is digest-linked to
its exact `ibm_om_deployment.pt` persistent state.

## Claim boundary

This implementation supports a model-based statement about IBM's published
AIHWKit OM fit under this controller. It is not raw trace replay, a physical
microSiemens calibration, endurance evidence, or proof that a fabricated
array can perform 128 writes with the simulated timing and energy. Network
training must not start if the cap-128 endpoint artifact is missing or
inadequate. On-chip recovery remains a separate, later hypothesis.

## Corrected fixed-array common-window pilot

The corrected two-arm study
[`mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2.json`](../studies/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2.json)
completed on 2026-08-24 and is artifact-verified as `ready_for_review`. It
tests one fixed counterfactually repaired OM array, not the published-corrupt
or held-out-array cases. Both arms use the same assignment, four-cell
dual-rail grouping, pulse-resolved selection stream, and apparent-forward
endpoint policy. The HWA arm is the only arm whose minibatch forward pass uses
the fixed-array compact endpoint model.

For each logical synapse, the mapper intersects the four assigned device
ranges and places the clean nominal targets in the central 50 percent of that
common range. The fixed assignment has 39,700 quads. Four W1 quads have an
empty intersection; all other quads have zero mapped cells outside their own
bounds. Compact HWA therefore uses fitted endpoints for 158,792 cells and the
predeclared exact pulse controller for the eight cells in those four empty
quads. All eight fallback cells programmed successfully, with 10.125 mean,
7 median, and 26 maximum pulses; none exhausted the budget or became
non-finite. These fallback statistics describe the final compact-HWA
realization recorded with the selected checkpoint, not an aggregate over all
minibatches.

| Training arm | Selected epoch | Clean accuracy | Pulse-deployed accuracy | Teacher agreement | Deployment KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | initial (-1) | 93.58% | 51.80% | 53.20% | 0.054706 |
| fixed-array HWA | 10 (stored index 9) | 86.64% | 62.98% | 64.60% | 0.054275 |

Relative to the clean-training deployment control, fixed-array HWA improves
apparent-forward accuracy by 11.18 percentage points and teacher agreement by
11.40 points. It reduces KL by 0.000431, or 0.787 percent, while reducing
clean logical accuracy by 6.94 points. The best physical accuracy during HWA
was 63.54 percent after epoch 6, but the predeclared KL selector chose the
final epoch. Thus HWA closes 26.8 percent of the initial 41.78-point
teacher-to-deployment accuracy gap and leaves a 30.60-point gap.

The matched deployment programming statistics are almost unchanged: the
clean control accepts 157,673 of 158,800 cells (99.290 percent), while the HWA
checkpoint accepts 157,657 (99.280 percent). The selected HWA deployment uses
21.063 mean and 11 median pulses, with 1,143 cells exhausting the 128-pulse
budget and no non-finite endpoints. Accepted apparent residual RMSE is
0.01369, versus 0.05655 for the hidden persistent state. Forward inference
uses the apparent endpoint; both states and the RNG continuation are retained
in the deployment sidecar for a later two-state update experiment. A recovery
run must explicitly load that sidecar, verify its selected-weight digest, and
prove the two-state restore/update semantics; the generic deployment field is
not the handoff.

This result supports the narrow conclusion that, with quad common-window
placement held fixed, exact fixed-array HWA materially improves deployment
accuracy on this one repaired array, but does not close the gap. It does not
establish that
on-chip training is required: that claim needs a predeclared matched recovery
arm starting from this exact two-state deployment bundle. It also does not
establish generalization to new assignments, behavior with published corrupt
devices, donor realism, or measured silicon beyond the AIHWKit OM fit.

## Predeclared canonical eight-device differential-pair pilot

- Status: interrupted partial evidence; the clean arm completed, while the
  HWA arm was stopped by user request after seven of ten epochs and is not
  eligible for workflow finalization
- Study: [`mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1.json`](../studies/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1.json)
- Claim boundary: an internally matched eight-device clean-versus-HWA test,
  not a matched comparison with the completed four-device v2 study

The next pilot uses the canonical `differential` DRN representation: every one
of the four signed dual-rail edges has a local `conductance_plus` and
`conductance_minus` cell, for eight memristors per logical weight. It does not
copy the four-device physical conductances into a different encoding. Both
arms instead load one tensor-identical current-schema canonicalization of the
existing clean ideal-differential checkpoint:

```text
source base:
  results/mnist-relu-drn-kd-exploratory-20260816/
  ideal_differential_finetune/
  20260816T135214.689196Z-9ef10aa2-642ca991/checkpoints/weights.pt
SHA-256:
  5a9dece30a938f00d2022de3e0aa57755e17f754c2df249507dd709bf9394fbc

ReLU teacher:
  results/mnist-relu-drn-kd-exploratory-20260816/
  teacher_fixed_init/
  20260816T132557.806720Z-fbff3c26-6f6867f1/checkpoints/weights.pt
SHA-256:
  9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52

IBM OM device model SHA-256:
  3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3
```

The source selected epoch 19 under the signed ReLU teacher. Its recorded
validation accuracy is 97.16 percent, teacher agreement 99.84 percent, and KL
0.000685286. Canonicalization is metadata-only: the launcher must prove exact
equality of all four named conductance tensors, validate current metadata, and
replay the source and canonical checkpoint identically before either arm may
start. It must not invoke teacher mapping, conductance remapping, retraining,
or an optimizer update.

The two arms then perform ten continuation epochs with the same ReLU teacher,
seed `42`, data split, solver, ideal optimizer, differential learning rates,
fixed base checkpoint, and pulse-resolved checkpoint selection. The clean arm
uses no minibatch modifier. The pair-HWA arm uses the compact OM endpoint
modifier on every minibatch. Both selection paths use the same exact physical
deployment.

The physical protocol is fixed before observation:

| Item | Declared value |
| --- | --- |
| Assignment / compact endpoint / pulse endpoint seeds | `84001 / 84002 / 84003` |
| Corruption | `counterfactual_repaired` |
| Target mapping | `differential_pair_common_window` |
| Pairing | canonical adjacent plus/minus cells at the same coordinate |
| Common-window margin | `0.25` per side, the central 50 percent |
| Start/controller/budget | lower-from-RESET, adaptive, 128 pulses |
| Endpoint use | apparent forward, persistent hidden update state |
| Target support policy | fail closed with `target_out_of_support=error` |

The fixed seed-`84001` integrity fixture has population fingerprint
`d0dfae135fa7b741c46f1178e52f645da4593f7313b4e6a0a71a9b3793d1b2f0`:
317,600 cells, 158,800 two-cell pairs, zero active corrupt cells, and eight
empty pair intersections. The authoritative mapper must show zero unsupported
cells in every nonempty pair and strictly positive nonempty inner spans. The
eight empty pairs account for exactly 16 out-of-bound cells. Compact HWA uses
317,584 fitted endpoints and the exact pulse controller for those 16 cells,
and only those cells, under policy
`pulse_resolved_noncorrupt_out_of_bound_empty_pair_only`.

Before native launch, the exact pulse canary must program all cells with at
least 99 percent success and at most one percent budget exhaustion. In its one
already-programmed context, the production evaluator processes the first 100
ordered validation batches—1,600 examples—and must retain at least 25 percent
student accuracy and teacher agreement. Strict JSON/checkpoint round trips,
durable compact and pulse deployments, and population/report/hash parity are
also launch gates. After both runs, the launcher audits the two result files,
population artifacts and receipts, mapping and programming reports,
checkpoints, and deployment bundles before marking the study ready for review.

Once the implementation and focused tests are reviewed, the formal workflow
is:

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1.json \
  --results-root results

python -m experiments.mnist_relu_drn.ibm_om_differential_pair_pilot_launcher \
  --aihwkit-python /home/filip/miniconda3/envs/aihwkit/bin/python
```

### Interrupted v1 observations

The formal launcher passed every declared preflight and completed the clean
arm. The HWA arm emitted seven durable epoch records through epoch index `6`
(`24,066` updates) before a user-requested stop. The worker exited by SIGTERM;
there was no numerical error, but there is consequently no terminal HWA
`result.json`, selected deployment sidecar, or post-run artifact audit. These
measurements are partial evidence and must not be entered into the finalized
study ledger.

| State | Accuracy |
| --- | ---: |
| Clean initial network | 97.16% |
| Clean-control apparent programmed deployment | 49.94% |
| Partial HWA KL-selected clean checkpoint, epoch index 5 | 95.72% |
| Exact pair-window mapping of that selected checkpoint | 95.16% |
| Apparent programmed validation at selected epoch index 5 | 55.72% |
| Post-hoc best apparent accuracy, epoch index 6 | 60.36% |

The exact selected-checkpoint mapping replay used the saved seed-`84001`
population and the ordered 5,000-example validation cohort. Mapping changed
122 predictions (2.44%) and cost only 0.56 accuracy points, but compressed
calibrated score RMS from `10.25447` to `0.008294`, approximately 1,236-fold.
The subsequent selected pulse evaluation lost 39.44 points relative to the
ideal mapped state. The declared KL selector retained epoch index `5` even
though epoch index `6` had 4.64 points higher accuracy; its KL was worse by
only `0.0000868`.

A read-only tensor decomposition of the completed clean-control deployment
isolates why the programmed signal is fragile. Pair-window mapping reduced
the per-edge plus/minus difference RMS from `0.13669` to `0.04975` in W1 and
from `0.03746` to `0.01377` in W2. At the same time, the normalized pair-sum
mean rose from `0.10422` to `0.74341` in W1 and from `0.03043` to `0.71689` in
W2. The shared baseline cancels algebraically in `G+ - G-`, but it remains as
passive conductance loading. Apparent programming then produced logical
contrast SNR `2.97` (`9.46 dB`) in W1 and `0.882` (`-1.09 dB`) in W2, with
8.44% and 22.0% sign flips respectively. Accepted per-cell residual RMS was
approximately `0.01367` normalized (`1.50` microSiemens), while 2,292 total
budget failures supplied a much larger tail.

The reproducible read-only receipts are:

```text
results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/
  analysis/ideal_pair_window_mapped_selected_epoch5.json
  analysis/deployment_decomposition_clean_signal_noise.json
```

Do not pool this result with four-device v2 or label their difference an
eight-versus-four effect. The studies differ in teacher and base checkpoint,
encoding, conductance units, learning rates, physical population size and
fingerprint, and common-window grouping. A descriptive mechanistic table is
permitted only with those mismatches visible. This eight-device pilot has no
Tiki-Taka, LoRA, or soft SET/RESET arm; any claim that on-chip recovery is
needed still requires a later matched study starting from one frozen
persistent deployment bundle.
