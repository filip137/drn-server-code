# Conv3 p95 read-noise first-level comparison

Approved by Filip with “Implement the plan” on September 21, 2026.
Six seed-0, thirty-epoch cases: baseline/ours/legacy at sigma 1e-3 first,
then the same three schemes at sigma 0. No further noise levels or seeds.

Use the September 19 refined strict per-matrix cosine >.95 selections:
injected beta baseline 147.682614594, ours 2.49274796756, legacy
2.81845428732. Preserve the exact parent base-beta conversions. The selection
covers every weight matrix on all 36 batches at initialization and BPTT-best,
with no norm gate or additional beta margin. Betas remain fixed with noise.

Inherit matched saved initialization, fresh Adam, exact scheme-specific weight
rates, zero frozen biases, weights [0,100], input gain 360, perfect diode,
float64 centered frozen-current EqProp, reset-each-batch states and T=K=8.
Ordinary MNIST fixed 55k/5k split, seed 0, batch 16/64. Noise seed 2026081601;
independent Gaussian perturbations of copied non-input +/- endpoint voltages
for gradient readout only. Relaxation and validation are clean. Official test
is disabled. Reuse the accepted operating point and preserve its finite-T
residual caveat. Existing H100 ten-epoch clean runs are admission evidence,
not controls for this thirty-epoch V100 comparison.

Scientific runtime is the unchanged parent source-continuation-v1 snapshot:
commit 022098cfe6eb151dff3f7403d3f4b358709152b9, archive
74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819.
Only new configs and transport wrapper are added; preserve unrelated edits.

Jean Zay: fmu@v100, gpu_p13, qos_gpu-dev, v100-16g, one GPU/four CPUs per
task, at most three tasks concurrent. Each case uses three ten-epoch chunks,
120 min allocations and 119 min payload timeout. Submit the next chunk only
for validated planned pauses; preserve Adam, weights, all RNGs, noise counts,
metric history and best checkpoints. Finish the noisy group before clean.
Maximum 18 production allocations/36 GPU-hours; 37 GPU-hours including
preparation. Expected 9–11 compute hours plus queues. First submission starts
a 48-hour overall deadline; each wave has a 24-hour scheduled-admission window.

Run local same-runner CPU functional smokes for all six configs (local CUDA
is unavailable), followed by one 30-minute V100 canary running all six cases
through three short chunks/two restores. The canary and production share the
same wrapper, module, source, configs, device and output filesystem; smoke
outputs remain separate. Validate semantic artifacts and Slurm resources.
No production before every gate passes. Check syntax, exact scope, explicit
index 0 selection, duplicate invocation guards and continuation preservation.

Before each armed attempt announce its immutable ID, receipts and deadline.
Verify scheduler truth and semantic progress immediately; publish a durable
heartbeat at least every 60s and check metrics/logs/GPU progress at most every
30min. Missing progress for 45min while running is an operational failure.
The first unexpected operational error stops dispatch and the current turn:
preserve evidence, perform bounded read-only diagnosis, report and wait for
new user authorization. No automatic retry, cancellation or repair.

A native NonFiniteTrainingError with matching failed run artifacts is an
explicit independent scientific outcome: retain error/epoch/batch, terminate
that case, do not resume it, and continue other declared cases. Never publish
a successful result.json for a failed case. A finite completed case passes the
inherited screen iff final validation is strictly less than 5pp below its own
best. Also report full curves, maximum temporary drawdown, losses and matched
clean-relative accuracy drop. A failed clean control prevents attributing its
paired noisy failure solely to noise. Missing/unverifiable coverage is
inconclusive; single-seed validation remains exploratory.

Local root: results/eqprop-conv3-p95-read-noise-1em3-20260921-v1/.
Remote root: /lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/eqprop-conv3-p95-read-noise-1em3-20260921-v1/.
Collect metadata first, then all terminal snapshots; validate locally,
register the physical inventory, and publish CSV and matplotlib trajectories.
Keep historical p90/p99 comparisons explicitly separate by hardware/runtime.

## September 21 user-directed continuation and conditional expansion

Filip authorized finishing all three sigma1e-3 cases through epoch30 and
explicitly removed the manual validation pause at each ten-epoch boundary.
This supersedes the earlier requirement to submit each chunk only after agent
checkpoint review: submit chunks1 and2 together, with Slurm aftercorr linking
each chunk2 task to its own successful chunk1 task. The unchanged native
continuation guards preserve training state. Review the results at epoch30.
Keep operational monitoring and the armed-attempt fail-stop policy. The old
monitoring attempt's SSH timeout receipt remains immutable; no training job
failed and all three original allocations completed the planned epoch10 pause.

Expansion is conditional on all three sigma1e-3 cases completing30 finite
epochs without numerical failure, with valid final artifacts. Epoch20 does not
release expansion. Add the inherited grid sigma={1e-5,3e-5,1e-4,3e-4,5e-4}
for all three schemes, seed0,30epochs each, with unchanged p95 betas and all
other scientific fields. Retain the three previously planned clean controls.
Use at most6 V100 GPUs concurrently across the study (3 for this immediate
continuation); maximum127 GPU-hours including the original37GPUh allowance.
The existing stability/clean-relative analysis and official-test prohibition
remain unchanged. Preserve the overall deadline2026-09-23T10:30:09.440114Z.
No additional noise level is submitted before the30-epoch gate passes.

## September 21 user-requested cancellation

Filip explicitly cancelled the baseline and legacy sigma1e-3 cases.
Slurm confirmed22683_0 and22683_2 CANCELLED at15:11UTC. Do not resume them.
Ours had already failed with NonFiniteTrainingError at epoch18/batch2046.
Preserve all partial outputs and failure/cancellation evidence. The original
all-three-finite30-epoch condition did not pass, so broad expansion remains
unreleased. Clean controls were not cancelled by this instruction and have
not started. The separately requested sigma5e-4 repeat is a new noise-level
request; no sigma5e-4 production has been submitted yet.
