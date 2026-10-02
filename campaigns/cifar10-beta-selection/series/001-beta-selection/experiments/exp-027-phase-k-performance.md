---
id: "exp-027"
title: "Mixed-precision EqProp speed and memory: packed phases and shorter K"
status: "complete"
hypotheses: ["H-021", "H-022"]
---
# Can we reduce the cost of float64 nudging?

User-authorized performance diagnostics for [H-021](../../../hypotheses/H-021-packed-nudged-phases.md)
and [H-022](../../../hypotheses/H-022-reduced-nudged-k.md). The existing five-epoch
runs remain stopped. Reference evidence is the
[precision/performance pilot](../../../../pilots/cifar-float64-performance-20261001.md).

## Matched comparison and decisions

| Arm | Nudged phases | Nudged K | Free T |
|---|---|---|---|
| reference | sequential | [6,6,4] | [6,6,4] |
| packed | concatenated +/- batch | [6,6,4] | [6,6,4] |
| reduced_k | sequential | [4,4,3] | [6,6,4] |

Ours V4/C1 CIFAR L8 blocks[3,3,2], batch32, seed0 initializer and fresh Adam
from the preceding V100 benchmark. Float32 free states, autograd/digital operations
and Adam; float64 nudged solves and centered gradient formula, cast back to32.
Keep existing convolution autograd work in all arms to isolate the interventions.
No read noise, beta recalibration, LR change, official test or full-epoch training.
Use identical deterministic augmented training prefixes and fixed initialization
betas[0.0005857333362799141,5.431465723731432e-5,2.204854001994856e-6].

Each arm uses a fresh process, same GPU allocation,35 updates with5 warmups then
30 measured. Report mean/median update time, phase timings, peak CUDA allocated
and reserved bytes, speed ratio and1407-update epoch compute projection. Preloaded
CPU batches exclude data loading; projection excludes validation/calibration and
is not a measured full epoch. Allocation and software identity must match.
Compare saved first-update convolution gradients on the identical checkpoint and
minibatch; later gradients are not matched because trajectories may diverge.
H-021 thresholds: >=5% faster, fits32GB, every convolution cosine>=0.99999 and
relative L2 error<=1e-4. H-022: >=10% faster, allocated peak<=1.05x reference,
plus a separately labelled initial cosine>=0.95/norm0.9–1.1 screen. No BPTT or
trained-checkpoint quality conclusion is implied by these performance measurements.

## Execution and monitoring handoff

Local root: `results/cifar-phase-k-performance-20261001-v1/`; all failed
and measured cells remain here. Remote root has the same basename under
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/`.
Source snapshot uses that basename plus`-source` under
`/lustre/fswork/projects/rech/fmu/ucy17uy/server_code/`.
Inputs and exact wrapper: `inputs/`, `launch/`; source identity is recorded there.
Target one Jean Zay V10032GB, fmu@v100, gpu_p13, qos_gpu-dev, one GPU,10CPUs,
module pytorch-gpu/py3/2.5.0. Total allocation cap30 GPU-minutes; the three short measured diagnostics run in one
20min allocation. The repository execution policy retires separate remote canaries
without a target-specific failure. Focused packed-solver tests precede these bounded
diagnostics; no full training batch is released.
Expected active compute5–10min plus queue waits. Stop on nonfinite, OOM, invalid
identity or budget exhaustion; retain failures without automatic batch-size changes.
A failed packed diagnostic does not block the independent reduced-K arm.

Monitor agent`/root/monitor_profiles` completed sole polling and collection
October1 at16:49 Paris; root owns review. Slurm458041_0 completed0:0 on r6i4n0
at16:46:48, elapsed6m09s, within the17:00:39 allocation deadline. All three arms
completed35/35 updates (5 warmup+30 measured); no failed or excluded arms.
Local bundles, Slurm stdout/stderr and accounting are collected; no next check is due.
The unchanged `analysis/analyze.py` passed using local py312: bundle validation,
source/config/checkpoint/environment matching, exact update coverage and initial
prediction/loss/force matching passed. `analysis/slurm-verification.json` passed
for one completed task with the declared scheduler contract and all three result
files; this verifies the bounded diagnostic, not a full training batch.
Measurements and gradient-screen statistics are in `analysis/summary.json`,
validation output in `analysis/validation.log`, accounting in `analysis/sacct.txt`,
and plot in `analysis/phase_k_performance.jpg`. Hypothesis verdicts are in the result note.
The first-update comparison is a numerical implementation/initial-K
screen, not a retired routine solver qualification or a training smoke.
Slurm singleton job **458041** started October1 at16:40:39 Paris on r6i4n0;
allocation deadline17:00:39 Paris. Monitoring is complete. Frozen-source CPU tests:24 passed; source
SHA256 identity, shell syntax and Slurm test-only checks passed.
Review and measurements
belong in `../results/exp-027-phase-k-performance.md`.

Reviewed outcome: packing preserves gradients but gives no speed gain; reduced K
saves17.14% update time and passes the declared initial gradient screen. See the
[result and limits](../results/exp-027-phase-k-performance.md). No further jobs queued.
