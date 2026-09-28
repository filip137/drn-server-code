---
id: "exp-010"
title: "Map initial layerwise cosine over normalized displacement and read noise"
status: "complete"
hypotheses: ["H-008"]
---
# exp-010 — Compact initialization cosine surface

## Question and dependency

Measure whether ours or baseline has an advantage anywhere in the declared grid.
Initialization only, as Filip confirmed September25. Depends on exp009's beta and
source lookup. Filip subsequently authorized this read-only replay on local/Akib/nom-cool-1.

## Cases and control computation

- Architectures Conv1/2/3; schemes baseline(1,1), ours(4,1), legacy(4,.25).
- Five output D/F targets:0.2,0.8,2,6,10. Use the same measured beta at every sigma.
- Noise sigma:0,1e-4,5e-4,1e-3,2e-3,5e-3; three independent nonzero-noise draws
  per batch, base seeds91000000/92000000/93000000. One clean comparison, no
  duplicate zero-noise draws. No extra initialization seeds in this first screen.
- Shared576-example validation cohort,36 ordered batches16; seed0 initializers,
  fixed preprocessing, paired20-output squared loss, T=K4/6/8, float64, explicit
  source diode parameters, zero frozen biases, [0,100] weights, gains40/100/360.
- Three convolution/readout weight tensors are not assumed: include2 parameter
  layers for Conv1,3 for Conv2,4 for Conv3; exclude frozen biases from cosines.

For each architecture/scheme/batch compute a clean post-T free state. From that
same state, BPTT differentiates exactly K further clean iterations using the same
loss/coordinates. Each beta's positive/negative phases use the common frozen
post-T teaching force. Cache BPTT (verify beta/noise invariance), free state and
clean endpoints; apply noise after relaxation before local energy-gradient formation.
Noise is independent by sign and noninput state layer, includes output, and never
perturbs clamped inputs, physical solves, BPTT or the displacement matching itself.
For equal-shaped tensors within a depth, pair the raw normal draws across schemes,
betas and sigma; record actual RNG keys/hashes and scale by sigma. Base-seed spacing
must keep all batch/phase/layer seed ranges disjoint between independent draws. Keep per-draw
cosines, not cosine of an average noisy gradient. Verify checkpoints/tensors unchanged.

This is270 architecture/scheme/target/noise cells,45 clean phase points ×36 batches
=1,620 clean physical batch replays. The declared full comparison count
is77,760 layer/draw/batch rows: (2+3+4) layers ×3 schemes ×5 targets ×36 batches
×(1 clean+5 noisy sigmas×3 draws). Save explicit coverage, avoiding duplicate clean
rows when a runner loops sigma. BPTT references are324 context/batch evaluations.

## Metrics and figures

For each parameter layer report mean, median and10th–90th batch/draw cosine;
paired differences versus legacy; gradient norms/RMS, noisy−clean gradient norm,
relative error to BPTT, absolute displacement D, normalized D/F, free-output RMS, and residual/zero-nudge
flags already emitted by the accepted replay. Undefined zero-norm cosines are
missing with counts, not zero/one. Do not apply a new T/K qualification sweep.

The region score is W=min_layer(mean individual cosine), as fixed in H008. Show
all per-layer results and all-layer dominance separately. No layerwise weight
norm rescaling or changing reference loss to improve apparent rankings.

Matplotlib PNG/PDF: per-depth panels for every convolution/readout layer, x=r,
y=nonzero sigma, absolute cosine and ours−legacy/baseline−legacy differences.
Show measured coordinates, clean reference curves separately, fixed[-1,1] cosine
and symmetric difference scales. Mask invalid/missing points; do not interpolate
color into claims about unsampled regions. Add worst-layer maps, absolute D/sigma
and raw B lookup so the normalization's unequal absolute SNR remains visible.

## Decision, budget and execution

Read H008's adjacent-cell candidate rule. Freeze and report the full grid even if
one cell favors a competitor; no early stopping at a preferred answer. No candidates
on a valid complete grid ends this screen with a scoped negative result. Candidates
trigger only the separately planned exp011 confirmation, never new training.

Proposed budget4 GPUh including smoke, calibration separate; measure one full case
before committing. Per the subsequent user assignment, parallelize whole architectures across local,
Akib and available nom-cool-1; keep all schemes for an architecture on one runtime.
Admit only if the sum of per-host allocated screen budgets fits4 GPUh; record
measured smoke extrapolation and individual ceilings before production.
Stop at cap, source mismatch or operational failure; retain numerical failures as
explicit cells and distinguish invalid coverage from a negative scientific result.

Planned existing study root: `results/conv123-initial-displacement-noise-map-20260925-v1/`,
child `screen/`. Directory creation and GPU execution are now authorized after the recorded gates. Save canonical
reporting bundles, resolved config/command/source/environment, state/gradient rows,
figures and a small report. Campaign note: `../results/exp-010-initial-cosine-surface.md`.

## Execution and monitoring handoff

Launch authorized September25; Codifier implementation/preparation is underway.
Prefer `experiments/replay_conv3_init_beta_noise.py`, which already reuses physical
endpoints/BPTT across a noise grid, and the existing all-depth gradient gate. The
initialization runner currently hardcodes Conv3/T8/gain360/one initializer/one noise
draw; adapt these explicitly for Conv1/2/3 and three draws without changing the
numerical estimator. Do not label the Conv3-specific CLI already compatible without
checking it. No new catalog or orchestration layer. Once launched, name one Monitor,
verify first finite artifacts and report every30min through a bounded deadline.

### Launch authorization and ownership

User assignment: "launch this; use local gpu, akibscomputer and nom-cool-1 if its
available". The series README holds the amended placement,6.5 GPUh total and
September25 20:00Paris wall deadline; immediate daytime execution is authorized.
Prepare owns scientific configs/runner/calibration; capacity owns staging/launch;
Medium monitor owns observation/collection; root owns analysis/interpretation.
No training is authorized. Record measured commands/hashes/handles below once real.

### Analysis implementation

`experiments/plot_initial_displacement_noise_map.py` consumes collected canonical
`layer_metrics.csv` and `state_signal.csv`, verifies complete36-batch/draw coverage,
explicit noise seeds, undefined cosine handling and45 actual output matches before
emitting confirmation candidates. It reports all810 layer cells, paired differences,
270 worst-layer summaries and pooled physical state/SNR controls. Matplotlib PNG/PDF
figures mask incomplete cells. Eight focused numerical/coverage tests and an empty-input
CLI/plot smoke pass. The helper additionally checks frozen config/runner identities, initializer/cohort,
paired noise/BPTT hashes, read-only guards and exact endpoint-draw coverage. The
first real Conv3 case passed this audit with2,304 layer rows. Interpretation remains
with the Reviewer after full collection.

## Execution and monitoring handoff — September25 source freeze

Scientific preparation passed45/45 local CUDA smoke contexts (2,160 layer rows);
runner hash cacca7852b8949164c65d19761ec7541acc3fd8ec18f0e77facae72131d06e0a.
The205-file numerical/config/input snapshot is `source-replay-v1.tar.gz` under
the registered study root, SHA256
666df5a434daad68074097ee526809d618a52f58d2298bf2c85eda857055daf1.
`SOURCE_SHA256SUMS` records every input; paths and bytes are preserved without
scientific config substitution. Source/init/proof hashes passed before freeze.

Placement: Conv3 localRTX3090; Conv1 thenConv2 serial on AkibRTX3080.
Nom-cool-1 is occupied by Ben on a3090 and is left untouched. All schemes,
targets and noises within an architecture stay on its recorded host/runtime.
Both hosts use py312/torch2.5.1/CUDA12.1.

Screen ceilings are1,800s Conv1,3,600s Conv2 and9,000s Conv3,total4GPUh,
including all preparation and target smokes. Local preparation elapsed36.21/
42.83/75.24s; the direct wrapper subtracts those and remote target-smoke time
from production allowances, counting localConv3's reused same-host smoke once.
Its unchanged smoke remains valid against the frozen runner/config hashes.
Conservative local smoke×36 estimates are22/26/46min; first full production
case timings will refine these estimates. Calibration used83.8s including its
parity smoke;0.5h matching and2h conditional confirmation ceilings remain
separate. Hard deadline September25 20:00Paris; daytime execution authorized.

Exact local command is `transport/local-conv3-command.txt`; direct bounded
launcher `transport/run-replay.sh` records PID,command,admission,elapsed and
exit receipts. Local outputs stay under study `screen/conv3`; Akib workspace
is `/home/filiposana/server_code/results/conv123-initial-displacement-noise-map-20260925-v1/replay_workspace`,
with source,transport,launch and `screen/conv1`, `screen/conv2` children.
Actual handles and semantic startup observations follow after admission.

### Actual launch handles and target gate — September25 11:51 Paris

Local Conv3 tmux `initial-cosine-conv3-20260925`, wrapper2484272 and CUDA
worker2484293, is running. First full baseline target0.2 context completed
in116.56s and its bundle validates; simple15-context projection is29min,
well within the2.5h architecture ceiling. An initial log-redirection attempt
failed before any worker/artifact because `launch/` did not exist; creating
that parent and relaunching started exactly one scientific worker.

Akib target CUDA smokes completed all30 contexts and were collected locally
with zero canonical validation errors. Runner/config/cohort identity and
BPTT/noise pairing invariants passed. Conv1/Conv2 measured smoke times were
10.83/18.02s (wrapper wall14/23s); conservative×36 estimates7/11min.
The finite Conv1→Conv2 queue is **555250**, launched with the exact command
in `transport/akib-command.txt`. Each architecture retains all15 contexts.
Its outer production timeouts subtract both local preparation and actual
target-smoke wall time:1,749s Conv1 and3,534s Conv2. LocalConv3 uses8,924s
after its75.24s same-host smoke. Total screen ceilings remain4GPUh.

Logs: local `launch/conv3-screen.log`; Akib `launch/conv1-screen.log` and
`launch/conv2-screen.log`. Per-architecture receipts are
`launch/convN/screen/{pid,command.txt,admission.txt,started_at,finished_at,exit_code}`.
Akib queue receipts are `launch/akib-screen-queue/`. Canonical cases are
`screen/convN/runs/convN_SCHEME_relativeTARGET/`; each updates status every6
of36 cohort batches, writes `artifacts/layer_metrics.csv` and successful-only
`result.json`. Summary `screen/convN/summary.json` must cover15 contexts and
17,280/25,920/34,560 layer rows for Conv1/2/3.

Medium accepted local monitoring with first-full-case check then15–30min;
Akib ownership follows first finite progress verification. Monitor only terminal
case artifacts for validation; do not hash active checkpoints. Collection:

```bash
rsync -a -e 'ssh -F /home/filip/.ssh/config -o BatchMode=yes' akibscomputer:/home/filiposana/server_code/results/conv123-initial-displacement-noise-map-20260925-v1/replay_workspace/screen/ results/conv123-initial-displacement-noise-map-20260925-v1/collected/akib/screen/
rsync -a -e 'ssh -F /home/filip/.ssh/config -o BatchMode=yes' akibscomputer:/home/filiposana/server_code/results/conv123-initial-displacement-noise-map-20260925-v1/replay_workspace/launch/ results/conv123-initial-displacement-noise-map-20260925-v1/collected/akib/launch/
/home/filip/miniconda3/envs/py312/bin/python -m experiments.reporting validate-run LOCAL_TERMINAL_RUN
```

Local Conv3 is already authoritative; Akib authoritative copies are under
`collected/akib/`. Codifier owns recovery/transport, Medium owns routine
monitoring and factual collection, and the assigned Reviewer evaluates H008
only after45/45 reconciled context coverage. Conditional confirmation remains
unlaunched until its predeclared candidate rule and separate gates pass.

Akib startup is verified: Conv1 wrapper555255/CUDA555280, two complete
36-batch baseline contexts and finite target2 progress. The target0.8 context
took22.7s, projecting roughly6min Conv1; Conv2 target-smoke timing projects
roughly10–11min. Medium explicitly accepted queue555250 and both architecture
allowances/deadline, with its next check in5min for Conv1 completion and Conv2
startup; localConv3 monitoring remains active. No duplicated polling owner.


## Terminal review

All45contexts/77,760comparisons validated locally September25. The screen is
complete; three H008 adjacent-pair candidates proceed to the already authorized
exp011. See [reviewed result](../results/exp-010-initial-cosine-surface.md).
