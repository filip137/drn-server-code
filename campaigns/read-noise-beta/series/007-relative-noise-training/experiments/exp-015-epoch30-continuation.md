---
id: "exp-015"
title: "Twenty more epochs from selected relative-noise epoch-10 weights"
status: "complete"
hypotheses: ["H-011"]
---
# Exp015 — Training to cumulative epoch 30 with an Adam reset

**Completed September 28:** all 20 retained cases are locally collected,
validated and reviewed. See the [combined result](../results/exp-015-epoch30-continuation.md#combined-review--september-28).
This supersedes the partial handoffs preserved below.

**Current scope:20 runs — original12 at eta1e-4,3e-4,1e-3 plus8 at eta1e-6,1e-5. The proposed1e-8/1e-7 continuation cases were withdrawn before training.**

Filip explicitly authorized 20 additional epochs from exp014 epoch-10 weights,
Conv2 initial output D/F=4 and Conv3 D/F=6, both ours and legacy at all three
relative noise levels eta=1e-4,3e-4,1e-3 (12 runs). He explicitly approved fresh
Adam state after being informed that optimizer/RNG state was not saved.

Result root registered before creation:
`results/conv23-relative-noise-epoch30-20260926-v1/`.
Configs: `configs/conv/relative_noise_epoch30_20260926/`.

Use each case's final epoch-10 weights, never independently chosen best weights.
Preserve parent learning-rate vector, beta, noise model and eta, dataset/split,
T/K, batch size, amplification and numerical implementation. Beta remains the
value matched at original initialization; do not recalibrate at epoch10.
Fresh Adam and reset deterministic RNG/data streams are explicit deviations from
uninterrupted training. Runner epochs1–20 map to cumulative epochs11–30.
Official test remains disabled. This is exploratory single-seed MNIST EqProp,
not CIFAR or cross-entropy training. Preserve parent checkpoint identity/hash.

Compare fixed cumulative epoch30 validation accuracy within each matched pair,
and compare with its epoch10 values. Retain all outcomes; selection is retrospective
based on exp014. No significance or global-optimum claim. Future checkpoints should
retain optimizer state; exact RNG continuation must not be asserted unless saved.

## Execution and monitoring handoff

New allowance60 GPU-hours including overhead/retries, deadline48hours from first
admission. Expected30–40 GPU-hours and8–14wallhours with parallel capacity;
contention may change estimates. Conv2 uses local/nom-cool-1 RTX3090; Conv3 uses
available trex/loulou/riri RTX5090, matched pairs on the same host. This is an
extension of exp014's authorized sharing policy: Ben/Kellian permitted with
measured memory plus headroom (Conv3 minimum8192MiB), other owners protected.
Retain one persistent scheduling/monitoring owner; never duplicate exp014 jobs.
No CPU fallback. User's no-tests/no-canaries instruction persists; verify source
checkpoint identity, CUDA admission, finite real training progress and output.

Source is exp014's frozen source-training-v1 archive. Parent outputs remain
immutable. Record exact queue/controller/watch handles here after launch. Collect
remote outputs locally, validate completed bundles and reconcile12cases before
final review. Primary result records live under this study root and this series.

### Launch and bounded recovery

Prepared11 terminal parents; Conv3 legacy D/F6 eta1e-3 remains dependent on its
exp014 final epoch10 checkpoint. Deterministic helper PID3696404 polls every300s,
checks terminal success, collects the final weights, verifies identity, and stages
its config/checkpoint locally and on riri; no best-checkpoint substitution.

The original loader constructed float32 parameters before loading float64 trained
checkpoints. All five initial attempts stopped before training; artifacts remain
under `failed-startup-dtype/` and their approximately5seconds each remain charged.
The minimal fix applies declared runtime dtype before strict loading only for
explicit weights-only continuations. Eleven restores were verified bit-exact; see
`restore-validation.json`. No rounding of saved weights. Frozen patched source
`source-training-v2.tar.gz` SHA256
`f337f482deb8a14614f5e8b8197613a06e365891d8d4ee2069810be29ffb8477`.
Configs and warmstart weights have separate identities in preparation.json.

Recovered controllers: local3697972, nom13480906, trex287493, loulou3457881,
riri333266. Exact commands and allocations: `transport/launches.json`.
Local handles eta1e-4 and1e-3 Conv2 pairs; nom1 handles3e-4 Conv2.
Conv3 eta1e-4 goes to loulou,3e-4 to trex,1e-3 to riri.
Local verified finite batch500 progress. Model and optimizer diagnostic checkpoints
are saved each epoch; these do not imply full private noise/data-order RNG restore.

Original hard deadline (not reset by recovery): 2026-09-28T18:38:28.889076+00:00.

After collection: `python -m experiments.analyze_relative_noise_epoch30 --study-root
results/conv23-relative-noise-epoch30-20260926-v1` with py312. Outputs coverage,
fixed epoch30 comparisons with epoch10, report and JPG under analysis/.
The independent monitor handoff contains current watcher ownership and observations.

### Lower-noise extension authorized September26

Filip requested adding lower noise levels. Add all four remaining original eta
values1e-8,1e-7,1e-6,1e-5 for Conv2 D/F4 and Conv3 D/F6, ours and legacy:
16additional runs,20additional epochs each, same explicitly approved fresh Adam
and reset RNG policy. Existing12 high-noise runs remain unchanged. Expanded
coverage is28continuations (14matched pairs) over all seven original noise levels.

Extension root registered before creation:
`results/conv23-relative-noise-epoch30-low-noise-20260926-v1/`.
Config root: `configs/conv/relative_noise_epoch30_low_noise_20260926/`.
Reuse the corrected frozen loader/source of the high-noise continuations.
Additional allowance80GPUh including overhead/retries, expected40–55GPUh;
extension deadline72h from admission, including waiting behind current jobs.
Existing60GPUh allowance/deadline is not changed. Prefer same five-host pattern,
with queues waiting for current authorized workers to finish. Expected completion
roughly18–30hours including waiting/contention, to revise from actual throughput.
No preemption, overlapping ownership, CPU fallback, or scientific retuning.
Monitor and review extension16 separately, then join the28 cases for final summary.

All16 low-noise parents are verified and staged. Detached extension controllers:
local3705597, nom13483714, trex290908, loulou3462856, riri338870.
They wait for the entire original high-noise lane to become terminal before
checking GPU capacity and admitting new work. Local runs Conv2 eta1e-8/1e-6;
nom1 Conv2 eta1e-7/1e-5; trex Conv3 eta1e-8/1e-5; loulou Conv3 eta1e-7;
riri Conv3 eta1e-6. Each placement retains its full matched scheme pair.
Extension hard deadline: 2026-09-29T18:54:41.070543+00:00.

Extension analysis uses `experiments.analyze_relative_noise_epoch30` with
`--config-subdir relative_noise_epoch30_low_noise_20260926 --expected-cases 16`
and the extension study root. Review original12 and extension16 separately; only
mark exp015 fully complete after both collections/coverage checks and combined
28-case review. Original12 completion alone is a partial study result.

### User correction: lower-noise extension narrowed to1e-6 and1e-5

This supersedes the earlier16-case lower-noise extension and28-case total.
Keep8lower-noise runs (both schemes, Conv2D/F4 and Conv3D/F6, eta1e-6/1e-5).
Remove1e-8/1e-7 from scheduling. All five extension controllers were verified
childless and waiting, with zero GPU time charged, before replacement. No
training jobs were interrupted. Preserve original proposals/receipts/configs in
extension `cancelled-scope/` and transport superseded files.

Active extension lanes: local Conv2eta1e-6, nom1 Conv2eta1e-5, trex Conv3eta1e-5,
riri Conv3eta1e-6, two cases each. Loulou extension lane withdrawn.
Budget reduced to40additionalGPUh (8/8/12/12h); original72h deadline unchanged.
Expected additional compute20–28GPUh, waiting on existing host work.
Analyzer uses `--expected-cases 8` and the same low-noise config subdirectory;
only8 active configs remain. Whole-study review requires20cases (12+8), not28.

### Original twelve collected and validated September27

All12 original cases are locally collected and validated, with exactly20 added epochs each; six matched pairs and zero validation/numerical failures. Charged outer runtime including startup retries is25.282559GPUh of60. Scoped evidence: [result note](../results/exp-015-epoch30-continuation.md), with detailed roots and receipts in original study `monitor/FINAL_SUMMARY.md`. Root Reviewer owns scientific interpretation; no REVIEW_COMPLETE is asserted. Whole20-case completion remains pending the independently owned eight-case extension and combined review. This incident agent did not inspect or act on that extension.

### Eight-case extension complete September27

All8 retained extension cases locally collected and validated with20 added epochs, verified final-parent10 identities and zero numerical/validation failures. Charged17.190655/40GPUh. Scoped interpretation and measurements are in the existing exp015 result note; extension monitor/FINAL_SUMMARY.md records validation. Extension REVIEW_COMPLETE is scoped to8 cases. Original12 sections/watcher remain unchanged; whole-study status stays partial pending combined root review.
