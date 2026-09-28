---
id: "exp-012"
title: "Compare absolute and nodewise relative endpoint noise in Conv3"
status: "complete"
hypotheses: ["H-009"]
---
# exp-012 — Compact relative endpoint read-noise replay

## User override: exploratory results first

On September 25, after the local smokes passed and Akib's clean state hashes
matched but BPTT gradient hashes differed, Filip instructed: "you can skip all
the checks, get me results asap." This supersedes the further parity/export
and qualification gates below. Local target2 proceeds with the frozen v4 runner;
Akib target6 uses a minimal revision that permits the old BPTT hash mismatch.
No additional smoke, parity export or numerical-equivalence gate is required.
GPU-only execution, memory limits, finite computation and exact-state conditions
for reusing old residual diagnostics remain in place. Report the comparison as
exploratory, with cross-device BPTT numerical equivalence unchecked. Each target
still compares all schemes on one host. Summarize completed outputs immediately;
do not withhold results for additional provenance checks.

Reporting caveat: the active v5 runner's aggregate `summary.json` retains an
inherited hardcoded `screen_reference_hashes_matched=true`. Do not interpret
that aggregate flag as a parity result. Per-run terminal metrics record the
actual bitwise match and the explicit waiver; those and the native references
govern the exploratory report. Preserve original artifacts, with this caveat,
rather than rerunning or modifying the active scientific source.

## Question and authorization

Does legacy retain its later-layer gradient-alignment advantage when read error
is proportional to voltage? Filip requested this short campaign and authorized
the local GPU and akibscomputer on September 25. The hypothesis is retrospective;
the following comparisons and decision rule are written before the new results.

## Cases and controls

- Conv3, seed-0 shared initializer; baseline (1,1), ours (4,1), legacy (4,0.25).
- Two matched initial output D/F targets: 2 and 6. Reuse exact scheme-specific
  injected betas and calibration proofs from
  `configs/conv/initial_displacement_noise_conv3_20260925.json`.
- Endpoint noise models: absolute, `v_tilde = v + sigma*xi`, with
  sigma=1e-4/1e-3 in simulator voltage units; nodewise relative,
  `v_tilde = v + eta*abs(v)*xi`, with eta=1e-4/1e-3 (0.01%/0.1%).
- One clean acquisition per context/batch, plus three paired draws at each
  positive level/model. Fresh base seeds: 101000000, 102000000, 103000000.
  Pair raw Gaussian draws across schemes, beta targets, models and levels;
  use independent nodes, phases and layers. Pure relative noise has no floor.
- All noninput endpoint reads, including output, are noisy; clamped inputs,
  physical relaxation and finite-K BPTT references remain exact/clean.
- Same 576 validation examples in 36 ordered batches of 16; deterministic
  55k/5k MNIST split, paired 20-output squared loss, float64, T=K=8,
  input gain 360, source explicit perfect-diode parameters, zero frozen biases,
  weights [0,100]. No optimizer, training accuracy or official-test evaluation.
- Six physical contexts, 13 acquisition draws per context/batch and four
  parameter layers: **11,232 layer/batch/draw comparisons** in production.

Absolute sigma and relative eta have different units. Their equal numerical
values are two small diagnostic grids, not a noise-power matching contract.
Within-model scheme comparisons are primary. Model-dependent changes in noise
severity and covariance remain limitations of causal attribution.

## Execution contract

Runner: `experiments/replay_relative_endpoint_noise.py`. Configs:
`configs/conv/relative_endpoint_noise_conv3_local_20260925.json` (D/F=2) and
`configs/conv/relative_endpoint_noise_conv3_akib_20260925.json` (D/F=6).
Each host receives every scheme and noise model for its target, so a scheme
comparison never crosses hosts. Preserve the same cohort, initializer and
scientific computation on both hosts. Record actual runtime/source identities.

Reuse the frozen numerical runtime and clean EqProp/BPTT computation from
series005. Change only endpoint acquisition. Require read-only state/parameter
guards, exact input reads, finite gradients, matched raw RNG hashes, source
identity checks, and measured pooled output D/F within 1% of its target.
Keep residual diagnostics at the accepted operating point; no extra T/K tests.

Require explicit CUDA availability and execution. Run a synchronous local
one-batch smoke through all declared cases and noise models for each config;
an Akib target smoke must establish actual memory fit before production.
An allocator/cache improvement that preserves tensors and arithmetic is allowed;
silently reducing batch size or changing the force/reference is not.
Do not overwrite old results or alter unrelated GPU workers.

Memory preparation: offload autograd's saved activation tensors to host memory
and restore them to CUDA for backward arithmetic, checking the resulting BPTT
hashes against the accepted replay. This preserves batch-16 semantics. Record
allocated/reserved GPU memory and process footprint in the smoke. A short local
replay may share the GPU with the existing CIFAR training only with demonstrated
memory headroom; the existing training is left running and sharing is recorded.

September 25 memory accommodation, before new noise outcomes: the first two
local smokes stopped at the fixed 7.5 GiB allocator cap in residual-diagnostic
autograd, before BPTT/noise scoring (6.183s and 7.220s, both retained). Expanding
saved-tensor offload did not remove large live backward intermediates in that
diagnostic. Reuse the hash-pinned series005 residual rows for each identical
case/batch, conditional on matching reconstructed phase states, post-T state,
parameters, frozen force and numerical source. Record their source and reuse
explicitly. Physical relaxation, BPTT and noisy EqProp gradients still execute
unchanged on CUDA at batch16. This avoids redundant residual evaluation at the
already accepted operating point; it adds no solver test or new T/K choice.

## Measurements and decision rule

For every convolution and readout layer report mean per-batch/draw cosine to
clean finite-K BPTT, cosine to clean EqProp, clean EqProp/BPTT cosine, relative
gradient error, noisy-minus-clean norm, endpoint RMS, D/F, D/P, and expected and
realized noise RMS. P is the pooled positive/negative endpoint RMS. Keep
undefined cosines explicit. Summarize each independent draw across the fixed
cohort; three noise draws are not three model seeds.

The primary diagnostic layer is the third convolution, where the previous
absolute-noise advantage was clear. For each target/level, define the legacy
advantage over a competitor as mean cosine(legacy) minus mean cosine(competitor),
separately for absolute and relative noise. Report both BPTT alignment and
acquisition alignment to each scheme's clean EqProp gradient. Show all layers
and every case; do not select a new favorable point after inspection.

A reduction in legacy's advantage under relative noise, reproduced in all three
draws at both targets and both levels for the same competitor, supports the
predicted pattern. The sign check is a proposed exploratory rule, not a
significance test. No reduction over complete valid coverage argues against the
simple explanation in this grid; mixed signs or floor/ceiling effects are
inconclusive. A ranking reversal is not required. Even a consistent reduction
does not isolate a causal contribution from absolute-versus-relative noise
severity/covariance, establish a realistic hardware model, or predict training.

Generate matplotlib PNG/PDF comparisons by layer, with absolute and relative
panels labeled in their own units, plus a CSV and a concise report. Plot clean
references and noise-draw variation, not just network-averaged cosine.

## Budget, deadline and outputs

One GPU-hour total including smoke/retries; each target has a 1,800-second cap.
Expected duration 10–25 minutes per lane after admission, updated from measured
smoke throughput. Hard wall deadline: September 25, 16:00 Europe/Paris.
GPU memory fit is currently being checked on local RTX3090 and Akib RTX3080.
Queue an occupied lane rather than disturb its worker. Fail on CPU fallback,
source/coverage mismatch, nonfinite outputs or exhausted budget; retain failed
attempts. No unbounded retries, additional targets, training or confirmation
sweep are included.

Registered output root: `results/conv3-relative-endpoint-noise-20260925-v1/`;
local and Akib runs go under `local/` and `akib/`, with collected authoritative
remote bundles under `collected/akib/`. Save canonical manifests, statuses,
metrics and successful-only results. Collect and validate before interpretation.
Campaign result note: `../results/exp-012-relative-endpoint-noise.md`.

## Execution and monitoring handoff

Preparation: root owns the scientific contract and review; prepare owns the
runner/configs/tests; capacity owns read-only occupancy and transport preparation.
Live commands, source identities, smoke timing/memory, handles and monitoring
owner will be recorded here after successful gates. No production is claimed yet.

### Transport and first memory gate — September 25, 13:29 Paris

The immutable `source-v1.tar.gz` contains 208 hashed source/input files; archive
SHA256 `fe6f34ae201badc10867da96d3f5703a2af965cb69923a52f39e88eb9e3cd86d`.
Runner SHA256 `f340701fb3d3c4cbe0556fc73b99011be49c443461ed8e1a1e1938716d3f6837`.
It is staged and verified on Akib under
`/home/filiposana/server_code/results/conv3-relative-endpoint-noise-20260925-v1/replay_workspace`.
The first local smoke used the declared local config, CUDA, unchanged batch 16,
7.5 GiB allocator cap and saved-activation offload. Exact argv and memory samples
are in `transport/smoke-local/{command.json,memory.csv,measurement.json,output.log}`.
It exited 1 after 6.183 seconds: endpoint-residual autograd requested 1.72 GiB
while 6.92 GiB was allocated, exceeding the allocator cap. Its measured process
peak was 7,408 MiB and minimum actual free memory was 4,192 MiB. The unrelated
CIFAR CUDA PID 2546244 was left running; no headroom guard was breached.
This failed operational attempt and source are preserved. No production and no
Akib GPU smoke have been admitted; preparation is extending equivalent saved
activation offload to the residual path before another bounded gate.

The failed smoke output originally at `local/` was moved intact to
`attempts/failed-smoke-v1-local/`; the original source-v1 archive and transport
logs remain unchanged. The canonical `local/` path is reserved for the repaired
source-v2 gate and successful production.

The second immutable archive `source-v2.tar.gz`, SHA256
`c47107a291233d22c6fdd873563ee81b54534b45b9ccd9cd87a3bf7a3322533b`,
extends saved-activation offload across physical replay and acquisition. It
failed the same endpoint-residual backward allocation after 7.220 seconds.
Process peak was 7,408 MiB, minimum actual free memory 4,178 MiB. Its former
`local/` output is preserved at `attempts/failed-smoke-v2-local/`; exact command,
traceback and sampled memory are under `transport/smoke-v2-local/`. Combined
failed preparation cost is 13.403 seconds. Both attempts preserved the unrelated
CIFAR worker. Production and Akib GPU smokes remain unstarted while the live
backward memory requirement is diagnosed; no batch-size or precision change
has been made.

The third archive `source-v3.tar.gz` (220 pinned files), SHA256
`2ae8e6437eb0e5cfaec3123a3927f35caa2049a16def9fe167ca7f3fdabc2cc7`,
reuses pinned residual tables after state/source verification. Its first smoke
passed the original failing diagnostic site but found a second residual
recomputation in `_phase_diagnostic` / `_phase_residual`, which exceeded the
same allocator cap. This attempt cost 9.277 seconds, with 7,418 MiB process peak
and 4,187 MiB minimum free memory. Former `local/` artifacts are retained at
`attempts/failed-smoke-v3-local/`, with exact transport logs unchanged. Total
failed preparation is 22.680 seconds; no production has started.

### v4 local gates and Akib reference mismatch

The v4 immutable archive SHA256 is
`9b2e6ee022e7b9fdcf59f377182328dda16072455376520c5865143b23d5fe8e`
(226 pinned files); runner SHA256
`7e5a7634a5b14d5d9532b47dd39e602e74a25997bfc2bf58b0e3a6cc04abf6e7`.
Both local config smokes passed all three contexts and 156 comparisons each,
with zero canonical validation errors and exact prior state/BPTT guards.
Target2 consumed 34.016 wall seconds (runner 29.374), target6 consumed 34.021
(runner 28.941). Both process peaks were 4,856 MiB; minimum global free memory
was 6,731 and 6,737 MiB, respectively, alongside unchanged CIFAR training.
Evidence is `transport/local-smokes-validation.json` and the measured receipts.

Akib's actual target6 smoke exited 1 after 6.360 seconds: live phase/state
checks passed, but the exact old-screen BPTT gradient hash differed. No OOM
occurred. Its output is retained remotely at
`replay_workspace/attempts/failed-smoke-v4-akib/`, and transport diagnostics
are collected under `collected/akib/transport/smoke-v4-akib/`. No reference
check was weakened. Total preparation charged so far is 97.077 seconds.
Production remains held pending resolution of this cross-device reference gate.

### Local production admitted — 13:46 Paris

Target2 runs from unchanged `source-v4` in tmux
`relative-endpoint-local-20260925`, wrapper PID2590468, CUDA PID2590481.
Exact command is in `launch/local/production.sh` and
`launch/local/production/command.json`. Output remains `local/`; sampled
GPU/process memory and logs are under `launch/local/production/`. The wrapper
has a 1,743-second remaining hard cap, charging 56.697 seconds of preceding
local-target preparation against its 1,800-second allowance. It continues
alongside the existing CIFAR worker with a 2-GiB minimum-free-memory guard.
At 13:47 Paris baseline had completed 8/36 batches and 416 finite comparisons.

Filip subsequently explicitly requested skipping further checks and obtaining
results immediately. Consequently Akib's additional cross-device BPTT parity
export, numerical comparison and extra smoke are waived. Its pending minimal
runner revision will preserve CUDA-only execution, allocator/headroom guards,
matched within-host schemes, unchanged physical/noise calculations and cheap
state identities used for residual reuse. Cross-device BPTT equivalence remains
unchecked; the Akib results must be labeled exploratory. No successful smoke
receipt is fabricated for the failed Akib attempt.

Akib's exploratory production uses `source-v5.tar.gz`, archive SHA256
`a1edea6f56883cce3433fd2c3efe33a7c1f958dee4580f0de9442abd701b177f`,
runner `9881168dddd6248de650741742e421824b14ae272637ff16a96b238bb895955c`,
and config `0e1cdd64a80bb88a239076885153ea69c313a50d5475f8b8b49227796cee0b1d`.
The revision explicitly records `unchecked_cross_device_user_override` for
BPTT-reference equivalence and skips the smoke prerequisite by explicit user
instruction. All other numerical calculations and residual-identity checks
remain intact. It charges 40.381 seconds of local target6 preparation plus
failed Akib smoke, leaving a 1,759-second production hard cap. Exact wrapper
is `launch/akib/production.sh`; its remote root is the staged `replay_workspace`,
output `akib/`, receipt/log/memory paths `launch/akib/production/`. Actual GPU
admission found an idle RTX3080 with 9,627 MiB free. Local production continues
from v4 without source or config changes.

Akib launched at 13:48:44 Paris, wrapper PID562599, CUDA PID562602. At
13:49:29 its baseline had completed 9/36 batches and 468 comparisons; actual
CUDA memory was about 4,782 MiB with 4,837 MiB free. Local baseline progressed
from 8 to 11 completed batches between 13:47:20 and 13:47:46. Both are healthy
under their respective hard caps. The Monitor owns the persistent combined
watch, bounded collection, coverage reporting and the delegated exploratory review;
capacity retains operational recovery authority without additional science.
The exact aggregate preparation charge is 97.077 seconds; production caps of
1,743 + 1,759 seconds keep the combined maximum below 3,600 GPU seconds.
The local target2 projection is about 15.5 minutes from live sharing pace;
Akib target6 initial pace is about 9–10 minutes. These are projections, not
completed runtime claims. Both output roots must be collected and reconciled
as six contexts / 11,232 comparisons before final interpretation.

### Terminal closeout

All six production contexts completed and are collected: 11,232 finite rows,
no missing production cases. The [result note](../results/exp-012-relative-endpoint-noise.md)
records the exploratory H-009 support, exact coverage, retained attempts and
explicit waived cross-device BPTT checks. Measured total cost is 0.42464 GPUh.
Plots and CSVs are in the study analysis directory; no further admissions.
