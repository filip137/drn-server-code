---
id: "exp-016"
title: "Noiseless Conv2 and Conv3 controls across three seeds"
status: "complete"
hypotheses: ["H-011"]
---
# Exp016 — Noiseless controls across three seeds

**Current scope:18 runs: baseline, ours and legacy × Conv2/Conv3 × seeds0/1/2;30epochs, no read noise.**

Filip requested no-noise Conv2/Conv3 runs with three seeds after the epoch30
relative-noise comparison. Execution interpretation: ours and legacy, seeds0,1,2,
Conv2 with the beta selected at initial D/F4 and Conv3 at initial D/F6,30epochs
from initialization,12runs. No clean epoch10 checkpoint exists, so these use
uninterrupted Adam; exp015 instead reset Adam at epoch10. Do not attribute that
cross-study difference solely to noise. No user response is required for the
stated routine default, unless he steers epochs/schemes before launch.

Set global nodewise relative acquisition eta=0 and absolute std=0. Keep all other
numerical/model/optimizer settings from exp014, including per-scheme LR vectors,
T/K, precision, diode parameters, frozen zero biases and batch sizes. Seeds vary
actual initialized weights AND training shuffle; paired schemes share identical
weights for each depth/seed. Keep deterministic55k/5k split seed0 and no official
test. Seeds are not just noise seeds (noise is disabled).

Freeze the seed0-selected beta across all seeds, as with learning rates. D/F4/6
are nominal reference labels calibrated at seed0, not claims of exact matching
for seeds1/2. Do not run new beta calibration or LR/TK audits. Seed0 reuses the
accepted initializer; other seeds use the same initializer recipe.

Primary: fixed epoch30 validation accuracy for each scheme/depth/seed, matched
ours-minus-legacy differences and mean/sample standard deviation across3seeds.
Epoch10 checkpoints/metrics are secondary and useful for the old10epoch evidence.
No best-epoch selection and no statistical-significance assertion from3seeds.

Result root registered before creation:
`results/conv23-noiseless-three-seed-20260927-v1/`.
Configs: `configs/conv/noiseless_three_seed_20260927/`.
Reuse corrected exp015 frozen numerical source; record initializer/source hashes.
Save model and optimizer checkpoints every epoch. No CPU training fallback.
User's exploratory no-tests/no-canaries instruction remains active; retain input
identity, actual CUDA/finite-progress verification and nonfinite guards.

## Execution and monitoring handoff

Allowance80GPUh including preparation/overhead/retries,72h deadline from first
admission. Expected35–55GPUh, approximately10–20wallhours with parallel eligible
capacity, dependent on sharing. Prefer local/nom1 for Conv2; verified5090s for
Conv3; complete matched scheme pairs on one host. This continues the same
relative-noise campaign's Ben/Kellian sharing permission and measured Conv3
8192MiB admission floor; protect other owners and current training workers.
Queues check live availability at admission and preserve previous runs.
Record actual commands,queue/watchhandles and source identity here after launch.
Collect locally, reconcile12cases, interpret against the fixed comparison and
write result note/ledger. No changes to exp015 scheduling or scientific settings.

### Production admission

All12configs and6initializers prepared; see preparation.json. Seed0 weights
match exp014 exactly. Seed1/2 use existing canonical wide-Kaiming seed assets,
verified against the same recipe and distinct tensor identities per depth.
The portable source uses inputs/initializers/convDEPTH_seedN.pt.

Controllers: local4047289, nom13717648, fifi621770, loulou3784440, riri728517.
Exact commands/paths in transport/launches.json. Local handles Conv2seed0/2,
nom1 Conv2seed1; fifi/loulou/riri handle Conv3seed0/1/2 respectively.
Each scheme pair remains on the same host. Current80GPUh allocations16/10/18/18/18.
Frozen corrected numerical source identity:
`f337f482deb8a14614f5e8b8197613a06e365891d8d4ee2069810be29ffb8477`;
configs and initializer identities are separately recorded in preparation.json.

Hard deadline 2026-09-30T09:31:53.253053+00:00.

Final analyzer: `python -m experiments.analyze_noiseless_three_seed --study-root results/conv23-noiseless-three-seed-20260927-v1` using py312. Writes12casecoverage, pairedseed effects, scheme means/sampleSD, report andJPG underanalysis/. Monitoring handoff records actual watcher PID and firstobservations.

## Baseline addition authorized September27

Filip explicitly requested baseline as well. Add baseline Conv2 referenceD/F4
and Conv3 referenceD/F6, seeds0/1/2,30epochs, eta0, matching the clean controls'
initializers, shuffle seeds and split. Reuse accepted exp013 seed0 baseline beta
and source baseline learning-rate vector; freeze these across seeds. No new
calibration/LR/TK sweeps. Baseline amplification differs by definition; retain
its declared V/I settings. Total clean study expands from12 to18runs.

Supplement root registered before creation:
`results/conv23-noiseless-baseline-three-seed-20260927-v1/`.
Configs: `configs/conv/noiseless_baseline_three_seed_20260927/`.
New allowance40GPUh including overhead/retries,72h deadline from first admission;
expected18–28GPUh, waiting behind current work where necessary. Existing12 budget,
queues and outputs remain unchanged. Place each baseline on its seed's existing
comparison host: Conv2seed0/2 local,seed1nom1; Conv3seed0fifi,seed1loulou,seed2riri.
Wait for those original lanes to finish before new GPU admission. Save per-epoch
model+optimizer checkpoints. Reuse corrected frozen numerical source.
Independent supplemental queues/watch own only6baseline runs. Existing12 may
complete and be reviewed as partial; only mark wholeexp016 complete after all18
have been locally collected/validated and three-way seed comparisons reviewed.

### Baseline preparation and scheduling

All6configs staged. Accepted baseline seed0 beta is3.0901706245678895 forConv2
(D/F3.9999994034),3.1924590902511554 forConv3(D/F6.0000000104), matching576examples.
Baseline V/I=1/1, T/K6/8; source baseline learning-rate vectors are retained.
No new calibration. Same6initializers as original cleancontrols, beta fixedacrossseeds.
Full source and matching identities: supplemental preparation.json.

Controllers local4172570,nom13778526,fifi680875,loulou3871596,riri826321.
Exactcommands/caps in supplemental transport/launches.json. All wait for their
original clean lane to finish beforeGPUcapacityadmission; no preemption.
Hard deadline 2026-09-30T14:26:33.548164+00:00.

Baseline analyzer: `python -m experiments.analyze_noiseless_three_seed --study-root results/conv23-noiseless-baseline-three-seed-20260927-v1 --config-subdir noiseless_baseline_three_seed_20260927 --expected-cases 6 --schemes baseline`. Collect/review original12 and baseline6 independently; combined three-wayreview requiresall18.

### Fifi queued work relocated at user request

Cleanbaseline seed0 movedfromFifitoTrex; controller545237. It startsimmediately without waitingforFifi originalcleanpair. Ours/legacyseed0 remainonFifi; baselinehostisnowTrex, a recordedplacement difference requestedbyFilip.
Onlyunstarted childlesscontrollerswerestopped;zerochargedGPUtime. OldFifireceiptsarepreservedlocally underrelocated-from-fifi; exactreplacementcommands intransport/launches.json. GPUverifiedTrexRTX5090 with31960MiBfree andBen-onlysharing. Numericalsource/configs/initializers unchanged;originalcaps andharddeadlinespreserved. ActiveFifitrainingnotinterrupted. Monitorlanes/collectionnowfollowTrex.

### Baseline terminal handoff

Baseline6 collected, validated and reviewed; see [scoped result](../results/exp-016-noiseless-three-seed.md). Charged18.843506/40GPUh. Original12 and combined18 review are not certified by this incident; overall status remains running pending its owning handoff. Baseline monitor may stop; no additional simulations authorized by completion.

CompletedSeptember28:all18collected/validatedandreconciled; combinedreviewintheexp016resultnote.
