---
id: "exp-020"
title: "Matched baseline continuation on available3090s with the exact5090 runtime"
status: "running"
hypotheses: ["H-010"]
---
# Advance the passing baseline candidate while5090 capacity is occupied

Current milestone: native epoch5 collected with7,035 updates,60.02% validation
accuracy and CE1.101680,7,624.0s for the continuation. EqProp1x has finished
all7,035 updates and validation at58.14%, CE1.185933; final collection passed.
Accuracy passes58.02%, but CE exceeds1.156764140 by2.52% (7.65% above native).
Filip stopped all new runs at09:00 Paris; collect this result but do not extend.
Evidence: root `analysis/native_bptt-5-collection.json` and
`analysis/beta_1-5-collection.json`; do not substitute
the original5090 native5 reference.

Baseline1x passed epoch1 in exp-013 (34.86%, CE1.754371 versus native27.92%,
CE2.095195), but its epoch5 continuation is waiting for an authorized5090.
Resume both native and1x from their exact saved epoch1 states on the two3090s.
Use a project-local byte copy of riri's Python3.12.13/torch2.11.0+cu128/
torchvision0.26.0+cu128/cuDNN91900 environment, not the older3090 runtime.
Both hosts support the build's sm_86/CUDA12.8 requirements. Verify copied
runtime identity and CPU imports; do not run a GPU smoke or migration audit.

Keep scientific source and inputs byte-identical to exp-013's baseline cohort.
Restore each branch's full epoch1 model/Adam/BN/RNG/scheduler state (step1407,
scheduler1, next batch0). Preserve all original rates, fixed betas, T/K,
batch32/eval16, seed0,45k/5k split/order/augmentation and50-epoch schedule.
Only the GPU architecture and transport paths change after epoch1. No new
epoch1, beta calibration or noise test. Runtime copying is isolated from
existing environments; no driver/MPS/GPU-wide changes.

The original native epoch5 on riri is retained as separate evidence. It must
not replace the new matched3090 native control in the gate. At5/10/20/30,
continue1x only if accuracy is within2pp and CE<=1.05x this matched control.
Continue exact full state through10/20/30/50 only after review. Failed quality
stops the branch. This cohort contains native and the already-selected1x;
the original unresolved0.3x screen remains in its original queue and record.

## Execution and monitoring handoff

Root: `results/cifar-eqprop-baseline-3090-continuation-20260930-v1/`.
Copy the runtime to a new project-owned results path on local and nom-cool-1;
expect5–15minutes per copy, then approximately2–4hours per continuation.
Prefer native on nom-cool-1 and EqProp on local after the short exp-019 replay.
Use the Ben-only owner guard and existing queue ownership.

Stage5:14,400seconds each. Transfer the queued original EqProp5 allowance
without duplication; hold/cancel it only after replacement preparation and
verification that it is still unstarted. Preserve the running/finished riri
native5 job. Fund the additional native4h by reducing unused baseline epoch50
caps16h→14h40m (57,600→52,800seconds) across at most three branches. The
combined baseline141.5GPU-hour ceiling remains unchanged. Subsequent declared
stage caps remain4h to10,8h to20,8h to30,14h40m to50; deadline October6 08Paris.

Keep new validated receipts/checkpoints in this separate cohort. Collection
must not overwrite original native5 evidence or invoke the original pending
0.3x initial round. Use an explicit collect-only transport path in the existing
wrapper; the coordinating reviewer owns milestone decisions using the stated
gate and prepared next-stage jobs. Shared queue owns operational monitoring,
recovery and collection; the existing30-minute summary includes this root.
Record handles, copied-runtime identity and actual restored progress here.


### Exact-state cohort and transferred local replacement

The isolated runtime is `runtime/py312/` beneath this cohort on both hosts.
The local copy passes CPU imports with Python3.12.13, torch2.11.0+cu128,
torchvision0.26.0+cu128, cuDNN91900 and NumPy2.4.4. All libraries load from
the copied prefix; the older3090 cuDNN preload/library overrides are cleared.
The original riri environment and existing3090 environments are unchanged.
`runtime/identity.json` pins runtime metadata and critical executable/library
hashes; `analysis/local-runtime-check.json` records the local inspection.

All original scientific input files remain byte-identical, including the
initializer and input identity embedded in both copied epoch1 checkpoints.
Their Adam steps1407, scheduler epoch1 and cursor next batch0 were verified.
All403 other source entries are byte-identical; the only changed entry is the
original cohort's wrapper with a minimal collect-only backport. The exact diff
is `analysis/collect-only-backport.diff`. It shares existing rsync/validation/
receipt behavior and skips advancement;26 maintained CPU transport tests pass.
No newer numerical helpers were copied into this older frozen baseline source.

Prepared stage5/10/20/30/50 jobs live in `launch/native_bptt-STAGE.json` and
`launch/beta_1-STAGE.json`. They invoke the original numeric training CLI with
the exact preceding full checkpoint. Native is nom-cool-1 only; EqProp is local
only. Both use the frozen Ben-only transport owner guard and16GiB+2GiB memory
admission. Callbacks use the copied runtime and frozen wrapper's `collect-only`;
all receipts stay in this separate cohort. `analysis/decision.json` explicitly
says `awaiting_review`; no original initial-round decisions are synthesized.

Local replacement ID: `cifar-eqprop-baseline-3090-e5-beta-1-20260930` was held
until the coordinating reviewer released it after exp019 completed. It began
training at06:10:41Paris on September30. A short future admission window
protected the initial submit-to-hold interval. The original5090 EqProp5 ID
`cifar-eqprop-baseline-e5-beta-1-20260929` was verified unstarted with zero
attempts, held, and cancelled only after its replacement was prepared and held.
Its14,400seconds transfer unchanged. `analysis/queue-transfer.json` records this.
The original riri native5 job/result was not changed or copied into the new gate.

Nine unsubmitted original baseline stage50 specs now cap at52,800seconds;
`analysis/budget-transfer.json` records the4h native-control transfer within
141.5GPU-hours. The shared half-hour summary includes this cohort. Future stages
remain unsubmitted until coordinator review. To submit an approved next stage,
use the copied Python from this cohort's `source/` and call the frozen wrapper's
`submit(stage, case)` function; it verifies/copies the actual parent checkpoint
and adds its hash before queue submission. Collection never calls this function.


### Actual restored startup, September30 06:15Paris

The riri-to-local runtime copy and CPU check finished before preparation. The
local-to-nom-cool7.6GB transfer ran approximately06:01–06:13Paris. The remote
CPU inspection verified all frozen source/input and prepared-runtime hashes,
training-data hashes, imports, and the full native epoch1 checkpoint before
submission. Evidence: `analysis/nom-cool-runtime-check.json`; no GPU smoke ran.

Both jobs are now running under the existing shared queue:

| Case | Queue ID | Target / PID | Attempt token | Observed restored progress |
| --- | --- | --- | --- | --- |
| Native | `cifar-eqprop-baseline-3090-e5-native-bptt-20260930` | nom-cool-1 /461028 | `ee7f9e651a6f4984ac4c06c54c631ff2` | update1460, epoch2 |
| Fixed1x EqProp | `cifar-eqprop-baseline-3090-e5-beta-1-20260930` | local /1605898 | `1a0fd7aaaf7348abb37dfefd236c85d1` | update1560, epoch2 |

Native started06:13:51Paris. Its transport owner check found only Ben; the
local check found no external GPU process. Both manifests report torch2.11.0+
cu128 and CUDA12.8, exact original parent checkpoint hashes, and the same
epoch2 training-order hash. Finite training losses and advancing cursors exceed
the restored1407 updates. `analysis/startup-proof.json` preserves these actual
artifacts and queue handles. The original5090 candidate5 is cancelled with zero
attempts, so it has no competing scheduler owner. The completed riri native5
remains separate evidence. Future10/20/30/50 jobs remain prepared only; both
completion callbacks collect/validate without advancing. The shared30-minute
summary includes this root and its explicit `awaiting_review` decision.
