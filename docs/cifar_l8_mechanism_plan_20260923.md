# CIFAR L8 amplification mechanism investigation

Authorized on 2026-09-23: implement the agreed causal-control plan, ten epochs
first. Original placement was Loulou; the user subsequently indicated Fifi is
free and clarified this is exploratory. All GPU work now runs sequentially
on Fifi, with the same cases and total budget. This is a seed-0 mechanism
diagnostic, not official-test or multi-seed accuracy evidence.

Cases, each fresh from initializer SHA256
`c4d98ef636949fed044ce4d2e94544be8c69eb816dc279338d7a067018fa6c37`:

| Case | Scheme | Complete optimizer vector | Block output normalization |
|---|---|---|---|
| baseline_reference | baseline | baseline_c1_d1_e10 | none |
| proposed_reference | ours | ours_edge_head_down_e10 | none |
| legacy_reference | legacy | legacy_c1_d1_e10 | none |
| legacy_baseline_lr | legacy | baseline_c1_d1_e10 | none |
| legacy_voltage_normalized | legacy | legacy_c1_d1_e10 | divide decoded outputs by16/16/4 before pooling/BN |

Copy the exact recorded optimizer dictionaries from the selected initial LR
study; do not round. Retain physical widths256/512/1024, blocks3/3/2, all
analog conv/classifier weights, zero analog biases, trainable affine BN and
boundary gains, Adam, batch32/evaluation16, crop/flip, cross-entropy without
label smoothing, conductance bounds[1e-7,10], T=K[6,6,4], and cosine
horizon50/final-ratio.02. Reuse the saved45k/5k split; never open test files.

The optional `block_output_normalization` field defaults to `none`.
`voltage` divides each decoded block output by A^(depth-1). It changes no
physical equation or checkpoint parameter key. Production BN epsilon remains
1e-5. Controls are opt-in; existing configs preserve their behavior.

Before production, run the existing amplification tests plus tests of the
new normalization setting, one end-to-end local smoke, and destination checks.
New normalization contracts must pass the existing scientific T/K gate:
reference[24,24,16], sentinel[48,48,32], and unchanged numerical thresholds.
Every production endpoint repeats its solver audit.

On Loulou, first perform200 matched training minibatches comparing baseline
with legacy using voltage normalization and the baseline optimizer. Use the
same epoch1 ordering/augmentation and full batch32. Logits, gradients, BN
buffers, and parameter updates must agree to relative error1e-5. A failed
gate stops admission and is diagnosed; it does not silently relax tolerance.

Checkpoint replay uses the actual saved initialization and epoch10/30/50
states for all three schemes. One deterministic cohort: first128 indices of
the saved training split, no augmentation, four batches32. Training-mode and
evaluation-mode results remain separate. Preserve checkpoint hashes and
parameters and restore BN buffers. Measure signed voltage mean/RMS/spread,
normalized voltages, diode occupancy, per-channel BN variance and epsilon
share, gradient RMS/sparsity, and relative proposed Adam updates. Compare
projected KKT and gradient agreement at operational/reference/sentinel
iterations. Distinguish raw voltage coefficients from normalized downstream
conductance loading. Matched-weight baseline/legacy replays isolate the scale
effect. Earlier and corrected physical-KCL MNIST evidence remain distinct.

Loulou's RTX5090/PyTorch2.11/cu128/cuDNN91900 is the sole GPU target. Check
all authorized hosts and configured Jean Zay before allocation, then wait
for Loulou's lane without interfering with unrelated jobs. Queue deadline24h;
total study cap15GPUh including GPU diagnostics, smokes and equivalence;
expected11–13h after allocation. Explicit CUDA residency is required; no CPU
training or saved-tensor offloading. Each ten-epoch case cap2.6h, with the
total budget authoritative. Do not extend training past10 automatically.

Register the result root before creation, maintain canonical run bundles,
source/config identities, logs, exit receipts and checkpoints. Watch semantic
progress every30minutes, diagnose missing/stale progress, collect remote
outputs locally and validate coverage before scientific closeout. Interpret
training CE and accuracy as well as validation; an LR-only intervention and
a normalization-only intervention answer different causal questions. A
ten-epoch result does not establish the cause of the final50-epoch gap.

Local and remote result suffix:
`results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/`.

Implementation clarification: full-batch128-example voltage/KKT replays and
operational gradients retain batch32. Long-unroll gradient checks retain the
existing eight-example/microbatch2 audit so reference/sentinel graphs fit on
the GPU without offload. These gradient comparisons have their own cohort and
batch labels, rather than being combined with the batch32 curve. Free and
tracked BPTT states are recorded separately. Local synthetic end-to-end tests
are followed by full-width, real-data destination smokes on Loulou.

Launch record: queued at2026-09-23 12:43UTC in Loulou tmux
`cifar-l8-mechanism-20260923`, launcher PID1508169. Exact command:
`/home/filip/miniconda3/envs/py312/bin/python -u /home/filip/server_code/results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/launch_loulou.py`.
The launcher waits for three idle observations before admission and checks
availability again between commands. Its queued heartbeat refreshes each
minute; running heartbeat each five seconds. State and terminal exit receipts
are under `launcher/`, subprocess logs under `logs/`, and run bundles under
`cells/` or the named diagnostic directory. Monitor resource, process, log,
and artifact progress every30minutes; stop and diagnose missing progress.

Local implementation checks passed146 tests with two CUDA-only skips. The
frozen source independently passed140 focused tests. All66 staged source
files and13 checkpoint files passed SHA256 verification before launch.
The real-data GPU smokes and200-step gate remain pending while Loulou is busy.

Placement update,2026-09-23 13:00UTC: Fifi is idle with the sameRTX5090,
PyTorch2.11/cu128/cuDNN91900. The Loulou queue was cancelled before any
computation (zero GPU budget consumed); no further Loulou checks are needed.
`launch_fifi.py` uses `source-fifi-v2`; the diagnostic target metadata is the
only source change. Source science, cases, cohorts and budget are unchanged.
The earlier Loulou launch is superseded. Fifi state is `launcher/fifi.json`.

Fifi launch:2026-09-23 13:03:34UTC, tmux `cifar-l8-mechanism-20260923`,
launcher PID2592831. GPU computation started13:04:34UTC. The two-minibatch
equivalence smoke passed with maximum relative error0; baseline preparation
and real-data smoke passed. State refresh and remaining work are in
`launcher/fifi.json`.

Parallel placement authorized2026-09-23: user explicitly requested splitting
the five cases between Fifi and Loulou, including use of free Loulou GPU
memory alongside its existing jobs. Keep baseline/proposed/legacy references
on Fifi; move legacy_baseline_lr and legacy_voltage_normalized to Loulou.
Reuse the completed Fifi qualification and smoke artifacts; no additional
Loulou diagnostic campaign. Preserve the active Fifi baseline worker while
replacing only its queue manager. Total cap remains15GPUh:7h for Fifi
(including already completed diagnostics) and8h for Loulou. Loulou controls
retain all scientific settings and ten epochs, with runtime-only caps4h
for shared-GPU execution. Expected elapsed completion depends on sharing;
measure actual batch pace after launch. Host is now a recorded exploratory
comparison limitation. State: launcher/fifi_parallel.json and
launcher/loulou_parallel.json.

Scope correction, 2026-09-23 15:17 UTC: user clarified that this is a BN
investigation and the legacy LR sweep and reference training already exist.
Reuse the original selected legacy ten-epoch run as the unnormalized control.
Cancel unstarted proposed/legacy repeats and stop the supplementary legacy
baseline-LR intervention; preserve its partial artifacts but exclude it from
the primary comparison. Preserve the nearly finished baseline repeat only as
auxiliary evidence. The sole new required training case is
legacy_voltage_normalized on Loulou: same selected legacy optimizer, same
initialization/split/order, ten epochs, batch32, trainable BN, augmentation,
and cross-entropy. Divide the decoded block outputs by16/16/4 before pooling
and BN, retaining BN epsilon1e-5. Reuse accepted smokes and solver qualification.
Expected duration about2.1h, hard cap4h, within the original15GPUh total cap.
Result remains cells/legacy_voltage_normalized in the same study root.
The previous five-new-case coverage requirement is superseded. No new LR search.

Completion18:27UTC: required normalized-legacy control is locally collected
and valid, all10epochs/Adam14070, solver audit pass,75.82% validation versus
74.68% original legacy. Runtime11294.81s within4h; sharing ended17:39UTC.
The older reference/LR repeat exclusions remain in force; no more jobs are
queued under this study. See findings for the supported ten-epoch mechanism
interpretation and its limits; the separate seven-case BN study remains active.
