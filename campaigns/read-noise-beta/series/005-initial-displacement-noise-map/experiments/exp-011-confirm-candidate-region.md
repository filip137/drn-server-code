---
id: "exp-011"
title: "Confirm a small initial cosine-advantage neighborhood with fresh noise"
status: "complete"
hypotheses: ["H-008"]
---
# exp-011 — Conditional confirmation, not another broad sweep

## Entry condition and selection

Only after exp010's full valid screen identifies an adjacent two-cell candidate
under H008. If none exists, record not-needed and do not launch this stage. Select
at most one pair per architecture and one competitor per pair using H008's fixed
largest-minimum-margin and tie rules; write actual selected r/sigma before measuring.
No beta/noise-grid expansion, trained checkpoint or training run is included.
A candidate from only a single layer/cell is reported but does not trigger this test.

## Cases and controls

At most3 architectures ×2 neighboring cells ×all3 schemes =18 scheme/cells,
with five fresh noise base seeds94000000/95000000/96000000/97000000/98000000 over the same36 batches. Use
exactly the same initialization, cohort, beta values, clean references, noise model,
T/K and gradient coordinates as exp010; only independent read-noise draws change.
Clean cached evidence is reused when raw tensors exist. The frozen screen runner
persists CSV/guards, not its in-memory endpoints/BPTT cache. Therefore confirmation
may reconstruct only the selected contexts with identical frozen numerical source,
beta, initialization and cohort, verify the clean gradient/state hashes against the
screen, then apply the five fresh draws. This implementation accommodation was
recorded before candidate selection; it changes compute reuse, not the scientific
comparison. All repeated selected clean solves count within the2 GPUh cap. Preserve
existing state flags; do not rerun the full screen.

For each independent draw compute each layer's mean cosine over36 batches, then
W=min_layer(mean cosine). Report competitor−legacy W and every per-layer difference
for both selected cells, all five draws. Support the candidate only if both W
contrasts are positive for every draw and all required comparisons are defined and
valid. Report mixed signs/undefined coverage as inconclusive, plus their direction;
never retry unfavorable random seeds. This is conditional descriptive confirmation,
not statistical significance across model initializations or a continuous-domain proof.

## Budget, outputs and next decision

Proposed cap2 GPUh including smoke; same recorded target/runtime as screen where
available. Stop if no candidate, budget exhaustion, source mismatch or invalid
execution; preserve all outcomes. Planned study child `confirmation/` under
`results/conv123-initial-displacement-noise-map-20260925-v1/`. Summarize validation
and H008 interpretation in `../results/exp-011-confirm-candidate-region.md`.

Only after confirmed initialization evidence, discuss whether selected existing
trained checkpoints are needed to test persistence. That would need a separately
frozen checkpoint-selection rule and new assignment; no new training is implied.

## Execution and monitoring handoff

Conditional execution authorized September25. No candidate cells are selected yet.
Future Codifier records the selected pair(s), exact replay command, identities,
target and deadline; Monitor collects/validates before Reviewer interprets H008.

### Launch authorization and ownership

User assignment: "launch this; use local gpu, akibscomputer and nom-cool-1 if its
available". The series README holds the amended placement,6.5 GPUh total and
September25 20:00Paris wall deadline; immediate daytime execution is authorized.
Prepare owns scientific configs/runner/calibration; capacity owns staging/launch;
Medium monitor owns observation/collection; root owns analysis/interpretation.
No training is authorized. Record measured commands/hashes/handles below once real.


## Selected cells frozen after the complete screen — September 25

All45 screen contexts/77,760 comparisons and45 actual output matches passed the
local scientific audit. H008 selected **ours** as the competitor in each depth:

| Architecture | First (D/F, sigma) | Second (D/F, sigma) | Screen W margins versus legacy |
|---|---|---|---|
| Conv1 | (0.8,0.001) | (0.8,0.002) | +0.322372607 / +0.372187745 |
| Conv2 | (6,0.0001) | (10,0.0001) | +0.144026453 / +0.204071378 |
| Conv3 | (0.2,0.002) | (0.2,0.005) | +0.000107823 / +0.000108791 |

The Conv3 margins are tiny and are not confirmation. The immutable selection is
`results/conv123-initial-displacement-noise-map-20260925-v1/analysis/candidate_regions.json`.
All three schemes remain included for each pair. New configs are
`configs/conv/confirmation_initial_displacement_conv{1,2,3}_20260925.json`;
new runner `experiments/replay_initial_displacement_confirmation.py` leaves the
screen source untouched. This requires12 clean contexts,18 noisy scheme/cells,
five fresh draws,9,720 noisy layer comparisons plus1,296 clean controls.

Selected-context reconstruction verifies screen post-T, zero, positive, negative,
frozen-force and parameter hashes; BPTT hashes and clean EqProp RMS must match
before adding fresh noise. Reference artifacts are copied with SHA256 guards into
study `confirmation/references/`. Same targets: Conv1/2 Akib, Conv3 local. Caps
900/1,800/4,500seconds total2GPUh include all local/target smokes. Existing
September25 20:00Paris deadline and numerical contracts remain unchanged.

## Execution and monitoring handoff — September25 12:23 Paris

Selected-pair local gates passed12/12 CUDA contexts,306 rows, with exact
screen state/BPTT hashes and five fresh draws. Separate source snapshot
`source-confirmation-v1.tar.gz` SHA256
e540c0cf3dc457bd8614be134704d02303814b778a51e75492cd23f7e79816b1
contains255 files; runner2525967925b75c566537ea3e0fae67921220f93444964b4c31a01a6f50e01529.
All source/config/checkpoint/proof/reference bytes are hash-checked; screen
source/results remain unchanged.

LocalConv3 tmux `initial-cosine-confirm-conv3-20260925`, wrapper2505414 and
CUDA2505435, passed fresh idle-GPU admission and reached6/36 finite batches
in its first baseline target0.2 context. Exact command:
`transport/local-confirmation-command.txt`. Log `launch/conv3-confirmation.log`;
receipts `launch/confirmation/conv3/confirmation/`; canonical output
`confirmation/conv3/runs/conv3_SCHEME_relative0p2/`,3 contexts/4,752 rows.

Same placement as screen: localConv3, serial Conv1 thenConv2 onAkib. Caps
900/1,800/4,500s total2GPUh include local and target smokes; local preparation
smokes elapsed7.414/15.546/14.135s. LocalConv3 uses4,485s production timeout,
counting its same-host smoke once. Akib outer timeouts will subtract local
preparation plus actual target-smoke wall time. DeadlineSeptember25 20:00Paris.

Akib workspace remains the screen's `replay_workspace`, with separate frozen
`source-confirmation-v1` and `confirmation/conv1`, `confirmation/conv2`.
Its target smokes precede production; actual handles follow. Expected total
12clean contexts,18scheme/noise cells,11,016 layer rows (9,720 noisy and
1,296 clean reference rows). No grid expansion or training occurs.

Medium monitoring requested after initial finite local progress; Codifier owns
transport/recovery. Inspect bounded logs and status every10–30min, collect
terminal Akib `confirmation/` into study `collected/akib/confirmation/`, and
validate canonical runs with py312 `-m experiments.reporting validate-run`.
Authoritative localConv3 is already under study `confirmation/conv3`.

Akib target confirmation smokes passed all9 selected contexts with exact screen
state/BPTT hashes and canonical validation; the authoritative local smoke copy
is under `collected/akib/confirmation/`. Runner elapsed2.13/5.38s, wrapper
wall7/9s. Production queue **558004** now runs Conv1 thenConv2, with outer
production limits885/1,775s after subtracting local preparation8/16s and
target smoke7/9s. Exact command `transport/akib-confirmation-command.txt`;
queue receipts `launch/akib-confirmation-queue/`, per-depth receipts
`launch/confirmation/convN/confirmation/`, logs
`launch/convN-confirmation.log`. Conv1 requires3contexts/2,376rows and
Conv2 requires6contexts/3,888rows. LocalConv3 monitoring was explicitly
accepted by Medium; Akib handoff follows finite startup verification.

```bash
rsync -a -e 'ssh -F /home/filip/.ssh/config -o BatchMode=yes' akibscomputer:/home/filiposana/server_code/results/conv123-initial-displacement-noise-map-20260925-v1/replay_workspace/confirmation/ results/conv123-initial-displacement-noise-map-20260925-v1/collected/akib/confirmation/
rsync -a -e 'ssh -F /home/filip/.ssh/config -o BatchMode=yes' akibscomputer:/home/filiposana/server_code/results/conv123-initial-displacement-noise-map-20260925-v1/replay_workspace/launch/ results/conv123-initial-displacement-noise-map-20260925-v1/collected/akib/launch/
```

Akib finite startup verified: Conv1 wrapper558009/CUDA558034, completed
baseline target0.8 and progressed ours target0.8 through24/36 batches.
Medium explicitly accepted queue558004, both architecture caps and deadline,
with a3min next check for terminalConv1/Conv3 and Conv2 startup. LocalConv3
first full confirmation context completed in97.73s and validates. All current
confirmation cases are admitted; no further experiment or grid expansion is
queued. Codifier remains available for operational recovery; Medium owns
collection/validation and the assigned Reviewer owns the H008 conclusion.


### Confirmation launch and monitoring handoff

All12 local CUDA smoke contexts passed306 comparisons with exact screen
trajectory/BPTT parity. Local smoke elapsed7.414/15.546/14.135seconds;9 Akib
target smoke contexts also passed and were collected/validated. Frozen runner
SHA256 `2525967925b75c566537ea3e0fae67921220f93444964b4c31a01a6f50e01529`;
separate confirmation source archive SHA256
`e540c0cf3dc457bd8614be134704d02303814b778a51e75492cd23f7e79816b1`.
The original screen runner/source remain unchanged.

Exact commands are study `transport/local-confirmation-command.txt` and
`transport/akib-confirmation-command.txt`; bounded wrapper `transport/run-confirmation.sh`.
Local Conv3 wrapper2505414/CUDA2505435; Akib queue558004 initially owns Conv1
wrapper558009/CUDA558034, then queues Conv2 serially. Transport/capacity owner:
relative_output_capacity. Monitor clean_beta_monitor acknowledged both lanes;
Reviewer/conditional Codifier relative_output_prepare owns collected review.
The same2GPUh aggregate cap includes all preparation/target smokes, with
900/1,800/4,500second architecture ceilings and20:00Paris wall deadline.

Local authoritative output is study `confirmation/conv3`; remote results collect
to `collected/akib/confirmation/{conv1,conv2}`. Expected terminal coverage is
12 canonical contexts/11,016 layer rows, including9,720 fresh-noise comparisons
and1,296 clean controls. Medium checks artifacts and collects terminal cases;
no candidate is called confirmed before full local validation. Source mismatch
or operational errors return to the Codifier; unfavorable draws are retained.


## Terminal review

All12contexts/11,016rows validated locally September25 before12:30Paris.
Conv1/2 each pass10/10selected cell/draw contrasts; Conv3 is mixed6/10 and
not confirmed. The scoped existential H008 claim is supported by Conv1/2;
see [reviewed result](../results/exp-011-confirm-candidate-region.md).
Approximately0.159GPUh including smokes, no failures/retries/exclusions; both
GPU lanes released. No further science is launched.
