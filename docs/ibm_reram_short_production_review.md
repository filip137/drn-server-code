# IBM ReRAM short-production review

- Review date: 2026-08-21
- Study: `ibm-reram-program-verify-noise-20260821-v2`
- Evidence tier: formal, model-based AIHWKit-preset characterization
- Review disposition: approved for local launch after the test and study-prepare gates
- Runtime objective: all four arms terminal within 24 hours of the first arm launch

## Decision

The exhaustive v1 design is superseded for the first endpoint-model milestone.
It crosses every one of 4,096 identities and eight repeats with 41 targets,
three tolerances, two starts, and two controllers: 16,121,856 trajectories per
arm and 64,487,424 total. That replication is not required to answer the
primary half-step-tolerance controller and endpoint-model question.

The v2 study retains the scientifically matched axes:

- optimized-material and HfO2 IBM presets;
- corruption disabled and the published corruption probability enabled;
- all 41 continuous targets;
- lower-to-target/SET and upper-to-target/RESET programming;
- one-pulse and adaptive controllers with matched programming streams;
- device-identity calibration, fit, and held-out validation partitions; and
- empirical, Gaussian, failure/cost, and Wan-2022 comparisons.

It fixes the intended deployment acceptance width at `tau/step=0.5`, uses
1,024 identities and four independent cycle repeats, and blocks boundary
conditioning once per device/repeat/start across targets. The immutable total
is 671,744 trajectories per arm and 2,686,976 total, a 24x reduction.

## Statistical review

Each condition/target contains 4,096 trajectories. Whole-device partitioning
provides 616 fit identities (2,464 trajectories) and 204 untouched validation
identities (816 trajectories). The fitted mean and log-standard-deviation
curves retain 41 target locations for a fourth-order model. The corrupt arms
are expected to contain roughly 138 OM and 100 HfO2 corrupt identities under
the published preset probabilities, with realized counts recorded rather than
assumed.

This is sufficient for the predeclared central coverage and normalized
Wasserstein gates, but it is not a high-confidence per-target extreme-tail
study. Four repeats sample cycle-to-cycle variation without pretending those
repeats are independent device identities. The resulting model is conditional
on the half-step tolerance and must not be used to claim tolerance
sensitivity. A later study must be declared if a different deployment
tolerance is selected.

## Conditioning review

The original eight-consecutive-quiet-pulse boundary rule dominated runtime.
In the 1,280-trajectory operational comparison it took a median 1,797.5
pulses, reached the 4,096-pulse cap at p95, and failed for 23.125% of cases.
Four quiet pulses at the unchanged `1e-6` span threshold took a median 138
pulses and p95 627.2, succeeded in every case, and left successful endpoints
at the sampled bound. Changing the threshold by 10x did not repair the
eight-pulse rule. V2 therefore changes the consecutive-quiet requirement,
not the boundary threshold.

The per-start blocked state is a simulated matched-design intervention. Each
repeat independently samples conditioning, then its immutable state is cloned
across targets and controllers. This removes target-dependent initialization
as a confound and reduces boundary work by 41x. The persisted-ledger integrity
gate proves that conditioning seed, success, pulse count, and apparent and
persistent starting states are identical across targets for every matched
device/repeat/start.

## CUDA and provenance review

The local host exposes one idle NVIDIA GeForce RTX 3090 with 24 GB memory.
PyTorch 2.5.1+cu121 in the `py312` environment can execute CUDA, while the
pinned AIHWKit 1.1.0 installation is in a separate CPU environment whose
CUDA-13 PyTorch build is incompatible with the host driver. Production
therefore uses a deliberate two-environment boundary:

1. `/home/filip/miniconda3/envs/aihwkit/bin/python` constructs the exact IBM
   device population and Wan reference and writes versioned, hashed receipts.
2. `/home/filip/miniconda3/envs/py312/bin/python` loads that population and
   evaluates the explicit pulse equation and controllers on CUDA.

Every trajectory retains an identity-derived programming seed. CUDA execution
generates each complete normal stream with an independently seeded CPU
generator and stages the immutable table on the GPU. This removes scalar RNG
dispatch from the pulse loop without sharing streams or making results depend
on traversal order. The backend is exactly replayable on the recorded stack,
but its vector-draw streams are not claimed byte-identical to the legacy
scalar-CPU v1 streams.

Review gates passed:

- buffered seeded rows reproduce their individually generated reference rows;
- buffered population mean and standard deviation pass fixed standard-normal
  checks;
- noiseless CPU and CUDA pulse equations are exactly equal for the focused
  transition sequence;
- CUDA state save/restore reproduces exact pulse continuation;
- two independent end-to-end CUDA smokes produced byte-identical population
  and trajectory-database SHA-256 digests; and
- the successful CUDA smoke produced every report artifact and passed all
  persisted-ledger checks.

## Runtime and storage review

The exact-width OM-continuous sizing gate used 1,024 identities, four repeats,
five targets, both starts, both controllers, the 512-pulse programming cap,
the 4,096-pulse conditioning cap, blocked per-start initialization, SQLite
event persistence, full integrity validation, endpoint fits, Wan comparison,
plots, and report generation.

It completed in 61.57 seconds with:

- 81,920 trajectories;
- 2,863,651 verify events;
- zero initialization failures;
- a passing integrity report;
- 1.07 GB maximum host resident memory; and
- a 236 MB trajectory database.

Linear target scaling gives 504.9 seconds (8.4 minutes) and approximately
1.94 GB per arm, or about 7.7 GB for four arms. The host has 779 GB free under
`/home`. A deliberately conservative 10x time factor is 84.2 minutes for a
parallel campaign or 5.61 hours if the four arms were forced to run
sequentially. Both satisfy the 24-hour gate with substantial margin.

## Launch and monitoring contract

The formal launcher is
`python -m experiments.reram_program_verify.local_short_launcher`. It starts
the four independent direct `python -m ebl characterize` commands on
`CUDA_VISIBLE_DEVICES=0` and does not replace the repository study lifecycle.
It fails closed unless the worktree is clean and records the exact source
commit in its launch contract before creating any native run.
Child-side live-ledger writes are deferred until all four native manifests
have captured the same frozen source identity; the launcher then refreshes the
repository-native `current_simulations.md` view on every heartbeat and at
termination.

- Study root: `results/ibm-reram-program-verify-noise-20260821-v2/`
- Native results: `runs/<arm-id>/<native-run-id>/`
- Persistent tmux handle: `reram_pv_v2`
- Launcher attempts: `launch/<UTC-attempt-id>/`
- Per-arm logs and PIDs: `<attempt>/<arm-id>.log` and
  `<attempt>/<arm-id>.launch.json`
- Heartbeat: `<attempt>/heartbeat.json`, refreshed every 15 seconds
- Terminal launcher receipt: `<attempt>/launcher_result.json`
- Study coverage: `analysis/summary.json` and `analysis/report.md`

Immediately after launch, every arm must have exactly one reviewed process,
a native `manifest.json`/`status.json`, and growing database or metric bytes.
The active tmux handle, four child PIDs, RTX 3090 memory/process occupancy,
native status, log tails, and heartbeat deltas are checked twice before the
launch is called healthy. Thereafter, artifact progress is checked at least
every 30 minutes. A missing process, nonzero exit, absent native artifact,
stale 45-minute heartbeat, or zero GPU worker occupancy while nominally
running is an operational failure.

The numerical runner does not support in-place resume. A failed operational
attempt is retained and may be retried only as a new native run under the same
arm after the cause is understood and the unchanged config/source guards pass.
No tolerance, seed, controller, pulse budget, or population setting may change
during recovery.

## Post-completion compact endpoint-model check

This analysis was added after v2 reached completion. It therefore does not
change the immutable v2 hypothesis or promote a post-hoc choice to
confirmatory evidence; the 128-pulse v3 successor predeclares the selected
candidate.

The accepted apparent residual is forced by verification into
`[-tau,+tau]`. A single uniform density over that window reproduced central
coverage but missed systematic target-conditioned mass. It passed 7 of 16
preset/corruption/controller/start conditions. The previously declared
fourth-order Gaussian passed none. An eight-bin piecewise-uniform density fit
separately at each target, with a `0.5` pseudocount per bin, passed all 16:

| Compact model | Adequate conditions | Normalized median target-binned Wasserstein |
| --- | ---: | ---: |
| Fourth-order Gaussian | 0 / 16 | failed gate in every condition |
| Global `Uniform(-tau,+tau)` | 7 / 16 | up to `0.1359` |
| Target-conditioned eight-bin uniform | 16 / 16 | `0.0323-0.0479` |

The selected model's pooled 90% coverage was `0.8978-0.9037` and its 95%
coverage was `0.9488-0.9511`. Failed non-corrupt endpoints and corrupt/stuck
endpoints remain separate empirical branches; neither is used to broaden the
accepted uniform density. Targets outside `[0,1]` are rejected by default,
and corruption is an identity-level mask that must persist across writes.

Non-corrupt bound variation is a second, distinct mixture. Pairing each
identity/repeat's persistent lower- and upper-conditioned states yields three
exact-target classes: below the lower bound, inside the interval, and above
the upper bound. Their fit-to-validation probability MAE was `0.88-1.53`
percentage points across populations, with a worst target/class error of
`5.31` points. At `x=0` and `x=1`, approximately half the devices cannot reach
the exact target. The half-step verify window reduces the structurally
unreachable mass to roughly `47-48%` for OM and `30-32%` for HfO2. Apparent
acceptance is modeled conditionally and does not change that latent class.

## Post-completion 128-pulse successor decision

This decision was made after v2 completed and therefore does not change the
v2 protocol, configurations, or interpretation. The completed validation
ledger showed that all one-pulse failures and 99.96% of adaptive failures at
the 512-pulse cap had target windows outside the paired persistent reachable
interval. Failed trajectories averaged only 0.03 reversals for one-pulse and
0.12 for adaptive control, so repeated overshoot correction was not the
dominant source of the long tail.

The table below is a retrospective prefix calculation on the v2 held-out
validation trajectories. `Success @128` counts endpoints first accepted by
pulse 128; pulse-work reduction replaces every observed count above 128 by
128. It is not an exact counterfactual adaptive replay because truncating an
adaptive batch at the new cap can add a final verify that was absent from the
512-pulse trajectory.

| Population | Controller | Success @128 | Success @512 | Estimated pulse-work reduction |
|---|---:|---:|---:|---:|
| OM continuous | adaptive | 87.03% | 91.54% | 48.68% |
| OM continuous | one-pulse | 93.85% | 95.44% | 46.46% |
| OM corrupt | adaptive | 79.27% | 84.03% | 58.65% |
| OM corrupt | one-pulse | 87.10% | 89.25% | 59.33% |
| HfO2 continuous | adaptive | 96.80% | 98.73% | 39.07% |
| HfO2 continuous | one-pulse | 99.16% | 99.74% | 21.97% |
| HfO2 corrupt | adaptive | 91.86% | 95.24% | 56.51% |
| HfO2 corrupt | one-pulse | 97.70% | 99.34% | 34.38% |

A 64-pulse cap was rejected as the first successor because it reduced OM
adaptive success to 77.83% without corruption and 70.78% with corruption.
Among validation endpoints accepted only after pulse 128, 7,021 of 9,746
adaptive cases and 3,315 of 3,989 one-pulse cases were outside the diagnostic
persistent reachable interval. The remaining reachable tail is why 128 is a
new comparison rather than being declared equivalent to 512 from prefix data.

The successor contract is
`studies/ibm-reram-program-verify-noise-20260822-v3.json`. It preserves the v2
population, seeds, partitions, targets, tolerance, starts, controllers,
conditioning, and analysis, while selecting the strict
`production_short_cap128` profile. Only an exact v3 rerun may provide the
128-pulse endpoint and failure models used for deployment.

After preparing that study, the reviewed launcher selects it explicitly:

```bash
python -m experiments.reram_program_verify.local_short_launcher \
  --study-profile v3-cap128 \
  --aihwkit-python /home/filip/miniconda3/envs/aihwkit/bin/python
```
