# CIFAR L8 three-scheme Adam learning-rate search

User requested independent LR selection for baseline v1/c1, proposed v4/c1,
and legacy v4/c0.25, using MNIST Conv1/Conv2 rates as approximate priors and
five- or ten-epoch evaluations with parallel execution.

Retain the established fully analog eight-convolution plus dense-classifier
L8, logical widths128/256/512, blocks[3,3,2], trainable affine BN, max-pooling,
positive trainable boundary gains initially100, zero analog biases,
conductance bounds[1e-7,10], crop/flip augmentation and cross-entropy.
Adam, seed0, batch32, unchanged BN LR.001/gain LR.00005, cosine horizon50.
Use the existing exact saved initializer and stratified45k/5k training split,
matched epoch ordering and per-example augmentation. No official test access.

The existing CIFAR legacy vector anchors the centers. For conv weights,
multiply it by the geometric mean of the scheme/legacy ratios across MNIST
Conv1 C0 and Conv2 C0/C1. For the dense head, use the geometric mean of the
Conv1 and Conv2 dense scheme/legacy ratios. The MNIST source is
configs/conv/perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json.
These are approximate transferred priors, not MNIST handoff claims for CIFAR.
Complete CIFAR vectors live in configs/cifar/lr_search_20260921/*.json.

Run every combination of conv and head multipliers{1/3,1,3}:27 fresh
five-epoch candidates. No outcome-based omission. Rank eligible arms within
each scheme by epoch5 validation accuracy, then lower CE, then smaller LR.
Eligibility requires finite full coverage and a passing final solver audit.
Repeat the top two independently from the same initial state for10epochs on
matched RTX5090/PyTorch2.11/cu128 hosts; select by epoch10 accuracy then CE.
An outer-grid winner is explicitly boundary-limited; if necessary, test one
additional factor-of-three point along each selected outer axis at5epochs
before final promotion (at most two extra arms/scheme). No unbounded search.
No50-epoch training or official-test evaluation is included.

Qualify each scheme independently at T=K[6,6,4], with[12,12,8] as declared
alternative, against[24,24,16] and reference stability[48,48,32]. Use the
same eight-example training cohort and all nine conductance-gradient gates
as the completed CIFAR study. Failing preparation blocks that scheme.
Every trained checkpoint repeats the fixed operating-point audit.

Distribute coarse candidates over Fifi/Loulou/Riri RTX5090 and available
Nom/local RTX3090. Keep all schemes represented on every participating host;
record per-arm host/software. Final10epoch ranking uses only the matched
5090 stack to avoid a scheme/GPU-type confound. Akib RTX3080 may participate
only if saved-tensor CPU offload passes numerical-equivalence and full-width
memory/throughput smoke at unchanged batch32; otherwise use it only for checks.
Offload changes storage, not microbatching, BN statistics or gradients.
No unrelated worker is stopped. MPS server alone is not a training worker.

Expected initial grid approximately30–40GPUh, about8–12hours with five or six
lanes; ten-epoch confirmations add approximately12–15GPUh. Initial hard
budget72GPUh including up to six boundary probes and checks;5h cap per
five-epoch candidate and5h per5090 ten-epoch confirmation. Re-estimate from
sustained timing before dispatching the complete queues. Stop admitting work
before the declared total budget is exceeded and report unresolved coverage.

Study root: results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1.
Use isolated source snapshots on every host, direct runner and simple per-host
command lists. Save canonical run bundles, exit receipts and complete configs.
Monitor to terminal state, collect and validate locally before conclusions.
Retain failed smokes/candidates with explicit exclusions. The repository's
current lightweight workflow authorizes direct scientific interpretation.

## Qualified launch placement

All three schemes pass T/K[6,6,4] with the same shared initial file. Source
snapshot `source` is preserved with preparation. Production `source-v2`
changes only optional saved-tensor storage and its regression test; model and
config bytes match. Every shared initializer SHA is unchanged from the prior
CIFAR study. All six destination30-batch smokes complete.

Smokes measure ~15.7s/30batches on the three5090s,35.7s on Nom,40.4s local.
The local cuDNN9.1 runtime is isolated under this result root; Nom reuses its
previous study-local cuDNN9.1 runtime. Fifi/Loulou/Riri all use2.11/cu128.
Riri's wall clock is roughly two hours behind other hosts; duration accounting
uses monotonic clocks and progress deltas, not cross-host timestamp ordering.

Pinned saved-tensor storage failed the strict gradient check because packing
made tensor views contiguous. The unpinned stride-preserving path passes exact
CPU/CUDA gradient comparisons on local and Akib. Akib's full-width batch32
smoke succeeds at1.75GiB but takes126.6s/30batches: over8hours projected for
fiveepochs, exceeding the5h per-case cap. It is excluded from production on
measured throughput; preserve all tests/smokes as operational evidence.

Five production lanes: Fifi7, Loulou7, Riri7, Nom3, local3. Each covers all
three schemes. Plain config lists are in configs/cifar/lr_search_20260921/lists.
Nine-hour hard limit per initial lane bounds initial production to45GPUh;
check actual accumulated receipts against the72GPUh overall limit before
admitting boundary probes or10epoch confirmations. Each arm has its own5h cap.
Expected core wall time7–9hours, about30–40GPUh. Final confirmations use the
matched5090 stack. No unrelated jobs or display processes are stopped.

The initial local smoke was rejected by automatic approval review because GPU
activity was unidentified. A read-only check established only desktop graphics,
no compute worker, and23GiB free. Review then approved the bounded smoke, which
completed. This evidence supports using the authorized local simulation lane.

Production launched around16:00UTC on all five lanes. Verified intended
workers and first semantic progress. Lane/worker PIDs: Fifi2269844/2269845,
Loulou522835/522836, Riri530299/530300, Nom1683892/1683893, local244801/244802.
Every lane has a9h monotonic deadline; per-case timeout5h. First100batches
advance on all three5090s. Selection tie-break is higher terminal accuracy,
lower CE, then lower conv multiplier, then lower dense multiplier.

Cross-host input audit passes: exact full45k sample-order hashes for epochs1–10
and exact augmented tensor/label hashes for the first64 examples of each epoch
match on all five production hosts. All six smoke bundles are collected and
validate locally. Qualification uses the previous shared initial SHA
c4d98ef636949fed044ce4d2e94544be8c69eb816dc279338d7a067018fa6c37.

At approximately 17:07 UTC, the three center-rate candidates are collected
and validate locally. Epoch-5 validation accuracy/CE: baseline 62.58%/1.0496,
proposed 59.84%/1.1265, legacy 65.62%/0.9596. All three final solver audits
pass; checkpoint finiteness, conductance bounds/updates and trained BN
scale/shift/count checks pass. These are partial measurements, not selected
learning rates. All five queues continue; 24 core candidates remain.

At 17:23 UTC Trex's RTX 5090 is newly idle (no compute worker; MPS server
only). Same-stack correctness smoke passes, inputs match all ten epochs, and
first-batch gradient norms match Loulou exactly. Sustained timing is slower:
100 batches take 115.06 seconds alone; two simultaneous 100-batch smokes take
169.85/170.95 seconds each and use 23.3 GiB total. This supports two concurrent
workers on Trex. No unrelated CPU or GPU job is changed.

Revised placement at 17:37 UTC transfers `baseline_c1_d0` from Riri,
`legacy_c0_d0` from Nom, and `ours_c2_d2` from local. Trex queue A runs baseline
then legacy (6.5-hour hard budget); queue B runs ours (4-hour hard budget).
Expected Trex duration is about 5.7 hours. Fifi/Loulou remain unchanged.
This removes slow tails from both 3090 queues. All 27 configs stay unchanged;
no extra scientific arm is added. Overall budget stays 72 GPUh; conservatively
count overlapping workers' native elapsed times separately for admissions.

Current training finishes uninterrupted. Old queue parents read their lists
once, so reserve each transferred output path on its old host. The existing
no-overwrite guard stops that parent after it records its current child's
exit. Preserve original receipts/logs and record the planned handoff. Restart
Riri's remaining four-arm list and local's remaining one-arm list under new
receipts, retaining the original 01:00 UTC deadline. Nom ends after its first
two arms. No scientific failure is inferred from the expected transport stop.
The initial idea of moving three fast-host arms serially to Trex was replaced
before launch on measured throughput; no old queue was changed for that idea.

Trex production starts at 17:39 UTC. Parent/first-worker PIDs are
1763430/1763433 and 1763456/1763457; both reach their first training batch.
The two-worker peak observed footprint is 23.3 GiB. The sequential and both
concurrent 100-batch smoke final model tensors are bitwise identical; all
smoke bundles are collected and validate locally. Exact revised coverage is
recorded in `placement-rebalanced.json` under the result root (27 unique arms).

Riri's first parent finishes its two scientific children with native exit 0
and reaches the planned no-overwrite stop. At 18:12 UTC it is recorded as
superseded; the remaining four cases start under `riri-rebalanced.json`,
parent PID 573329, with a 6.5-hour cap (within the original lane deadline).
No active training was killed or duplicated. Original receipts and the
expected transport exception are preserved.

Local's first child completes with native exit 0, then its original parent
reaches the same planned guard. At 18:30 UTC the remaining `legacy_c2_d2`
starts under `local-rebalanced.json`, parent PID 326396, with a 3.5-hour cap
and the same isolated cuDNN 9.1 runtime. The old local reservation is retained
under `reservations/local-ours_c2_d2`; the actual transferred output comes
from Trex. No active training is stopped.

The first local restart fails before its first optimizer step because its new
launcher omitted `CUBLAS_WORKSPACE_CONFIG=:4096:8`, required by deterministic
PyTorch 2.5.1 matmul. Preserve the failed canonical bundle, native exit 1
(3.52 seconds), and log in `operational_failures/local-rebalanced/`.
The transport now supplies the deterministic workspace default and records
the relevant environment. A two-batch/one-validation-batch same-config CUDA
regression smoke (`smoke-local-envfix`) passes. The unchanged production
candidate restarts at 18:35 UTC under `local-retry1.json`, parent PID 328782,
with the same 3.5-hour cap. Scientific source/config snapshots stay unchanged.
The failed attempt is operational evidence, excluded from LR ranking and
included in cost accounting. Existing PyTorch 2.11 Trex and rebalanced Riri
workers lack this explicit environment override but accept the enabled
determinism guard; Trex's sequential/concurrent equality checks pass. Use the
explicit setting for subsequent launches, including the confirmation stage.

At 21:19 UTC, 20/27 core candidates are collected and validated; all twenty
final solver audits pass. Local completes both assigned candidates, and Nom
completes both retained candidates then reaches the planned no-overwrite
guard. Their training PIDs have exited and GPUs are released. Nom
`launcher/nom-cool-1.json` is superseded, with the expected transport exception
and native successful exits preserved. Trex completes both concurrent first
cases and starts its remaining legacy case. Fifi/Loulou/Riri continue.
Current leaders remain baseline 62.58%, proposed 63.50%, legacy 65.62%; these
are partial five-epoch measurements, with no final LR selection yet.

At 21:47 UTC, Trex single-worker production sustains 799.39/800.72/801.31
seconds for its first three full epochs, close to the other RTX5090s. Its
PyTorch2.11/cu128 stack and exact input agreement are already verified.
Use all four matched RTX5090 hosts for confirmations when available, and
overlap independent schemes once their full nine-cell comparison and any
required boundary probes are collected and validated. This changes placement
only; no LR, cohort, initializer, promotion rule or study budget changes.

The 22:22 UTC pre-allocation inventory changes the available placement:
Nom now runs another user's GPU simulation, and Trex has a new other-user MPS
server plus queued GPU simulation launchers waiting on locks. These are left
untouched. Stage confirmations on Fifi/Loulou/Riri while those unrelated
queues occupy Nom/Trex; recheck availability before any later reassignment.
Trex's staged confirmation transport is unused at this point. The complete
inventory is retained in `confirmation_host_inventory.json`; Jean Zay shows
no current jobs, and Akib/local are idle but do not meet the declared matched
RTX5090 confirmation contract.

At 22:30 UTC, legacy has full locally validated9/9 core coverage and every
final solver audit passes. The interior center `legacy_c1_d1` wins at65.62%
(CE0.959595); runner-up `legacy_c2_d0` reaches63.26% (CE1.038759). No boundary
probe is required for its interior winner. Freeze fresh ten-epoch configs
`confirmation/legacy_c1_d1_e10.json` and `confirmation/legacy_c2_d0_e10.json`,
with unchanged initializer, input order, batch32, all Adam rates, affine BN,
augmentation, CE and50epoch cosine horizon. Riri runs both sequentially while
Fifi/Loulou finish the other schemes. Expected4.3h; hard lane budget5h.
Outputs remain under the existing root
`cells/{legacy_c1_d1_e10,legacy_c2_d0_e10}`. Run same-config two-batch smokes
first using the frozen scientific source and the explicit cuBLAS environment.
Promotion evidence and complete rates are in `analysis/legacy_promotion.json`.
Conservative admission accounting: completed native attempts37.3379GPUh,
remaining core allocation bound5.3626GPUh, failed startup0.0010GPUh, checks
reserve1GPUh, all six confirmations15GPUh, maximum six boundary probes9GPUh;
total bound67.7014GPUh, below72. Initial overlapping Trex worker times are
counted separately. Planned confirmation queue label is `legacy-confirm-riri`.

Legacy confirmation starts at approximately22:32UTC (agent clock), parent
656191, first worker656192. Both same-config smokes return native exit0 and
complete two optimizer steps on64 training examples plus16 validation
examples; collected bundles and exact configs validate locally. First full
training batch is observed and Riri's GPU is occupied only by the intended
worker. The new launcher records the explicit cuBLAS environment. Riri's
known wall-clock offset remains; use monotonic elapsed time for the5h cap.

At23:14UTC, Trex is available again: no compute process, no queued GPU
launcher or lock waiter remains, GPU0%/92MiB. Its completed core work and
checked software/input contract are unchanged. Use Trex for one next eligible
confirmation once its scheme grid is complete, subject to a fresh occupancy
check at launch. Keep Riri's already running legacy queue intact. Split the
baseline confirmation pair across Fifi/Trex if both remain available;
conservative3h per single-case lane adds1GPUh to the previous five-hour pair
reserve, leaving the maximum study admission bound below69GPUh.

At23:23UTC, all nine proposed-scheme core candidates are collected and
validated. The final candidate `ours_c2_d0` wins at65.84%/CE0.989811, versus
63.50% for `ours_c1_d0`. Its conv multiplier3/head multiplier1/3 corner
requires both declared one-step extensions. Run `ours_edge_conv_up` on Trex
(conv multiplier9, head1/3) and `ours_edge_head_down` on Loulou (conv3, head1/9)
in parallel, five epochs each,1.5h hard budget each, approximately1.1h each.
Full vectors are in `configs/cifar/lr_search_20260921/boundary/`; no BN/gain LR,
initialization, augmentation, split, optimizer, CE or solver setting changes.
Each uses an exact-config two-batch smoke and the accepted same-scheme
T/K[6,6,4] point; final checkpoint audits remain mandatory. Outputs are
`cells/ours_edge_conv_up` and `cells/ours_edge_head_down` in the existing root.
Use both probes before choosing two fresh ten-epoch proposed-scheme candidates;
no further edge expansion is allowed for this scheme. Selection evidence is
`analysis/ours_boundary_selection.json`. New inventory confirms Trex/Loulou
are free and only the intended workers occupy Fifi/Riri. This places Trex on
the edge probe first; its baseline confirmation may follow once it finishes.

At23:28UTC, all27 core candidates are collected and validate locally with
native exit0 and passing final solver audits. Baseline winner `baseline_c1_d1`
is interior:62.58%/CE1.049603. Runner-up `baseline_c0_d1` reaches61.28%/
CE1.099909. No baseline boundary probe is needed. Freeze fresh ten-epoch
configs `baseline_c1_d1_e10` on Fifi and `baseline_c0_d1_e10` on Trex after its
current edge probe. Each has a3h lane cap and expected2.2h duration. Retain all
learning rates, trainable affine BN, augmentation, CE, initializer/split and
50epoch cosine horizon; exact vectors and selection are in
`analysis/baseline_promotion.json`. Expected scientific coverage is35 runs:
27 core, the two triggered proposed-scheme edge probes, and six ten-epoch
confirmations. Only these two boundary probes trigger; no further expansion.
Core native time plus recovered startup is39.4477 conservative GPUh. Reserve
1h for checks,5h for the legacy confirmation queue,6h for baseline confirmations,
6h for proposed confirmations and3h for the two active edge probes:60.4477GPUh
maximum admitted study time, below72. Edge parents/workers are
Trex1834456/1834457 and Loulou650540/650541; both reach real training progress
after exact-config two-step smokes pass.

Baseline center confirmation starts at23:30UTC (agent clock), Fifi parent
2331284/worker2331285,3h cap. Its exact-config two-step smoke returns native
exit0 and semantic64train/16validation coverage; the intended full worker
reaches its first training batch. The second baseline config and list are
already staged on Trex but not launched while its boundary probe is active.

At00:34UTC Sep22, both proposed boundary probes are collected and validated;
all29 five-epoch candidates pass final solver audits. Head-down achieves
66.68%/CE0.941809; conv-up achieves54.26%/CE1.328089. Promote
`ours_edge_head_down_e10` (head5.334108501991406e-5) on Loulou immediately and
`ours_c2_d0_e10` (head1.6002325505974218e-4) on Fifi after its active baseline
confirmation. Both share conv multiplier3 relative to the proposed prior.
Full vectors and evidence are in `analysis/ours_promotion.json`; configs are
frozen under `confirmation/`. No further edge expansion is permitted; the
five-epoch head winner remains at the lowest tested head LR. Each fresh
ten-epoch run has a3h cap and expected2.2h duration, with unchanged affine BN,
augmentation, CE, shared initialization/split, solver and50epoch cosine.
All six confirmation configs now exist. Targeting Fifi for the second proposed
finalist avoids a longer sequential tail on Loulou.
Baseline second finalist starts on Trex at00:33UTC, parent1846641/worker1846642,
after its same-config smoke passes and the completed probe releases the GPU.
No current job is interrupted. All output stays in the existing result root.

Proposed head-down confirmation starts at00:36UTC (agent clock), Loulou
parent671281/worker671282,3h cap. Its exact-config smoke passes64train/16val
coverage and native exit0. The second proposed config/list is staged on Fifi
and remains pending until its current baseline confirmation completes. Both
new Loulou and Trex confirmation workers show real first-batch progress.

At00:42UTC, the first ten-epoch confirmation (`legacy_c1_d1_e10`) is collected
and validated:74.68%/CE0.717528, native exit0,14070 optimizer steps and passing
final solver audit. Its first five epoch learning metrics, gradient norms,
clipping and LR vectors exactly reproduce the original core run (local replay
comparison saved). Riri automatically starts `legacy_c2_d0_e10`, worker696991.
This is a partial1/6 confirmation result; no final scheme recommendation yet.

At00:46UTC, current remote manifest/status files are mirrored into local
active-run directories and the standard generated Active dashboard is
refreshed. These operational mirrors are incomplete bundles until terminal
collection; scientific aggregation skips running statuses. The full local
copy remains mandatory before selection/review. Refresh these live metadata
mirrors with each subsequent resource check and after phase transitions.

At01:42UTC, Fifi's baseline center ten-epoch run is complete and subsequently
collected and validated:75.26%/CE0.722593, native exit0,14070 optimizer steps,
passing final solver audit. The final proposed candidate `ours_c2_d0_e10`
passes its exact-config two-step smoke and starts on the released Fifi GPU,
parent2349651/worker2349652. It has the previously declared3h cap and expected
2.2h duration; no remaining scientific case is unstarted. All four5090s show
real training progress, approximately95–98% GPU utilization and11.9GiB each.
The full resource inventory, including idle Jean Zay, was rechecked first;
no unrelated jobs were changed. Local coverage is29/29 five-epoch plus2/6
ten-epoch cases, with all completed final solver audits passing. The last
Fifi smoke bundle and launch provenance are collected and validate locally.

At02:43UTC, proposed head-down confirmation on Loulou is collected and
validated:75.36%/CE0.691127, native exit0,14070 steps and a passing final
solver audit. This completes3/6 confirmations; Fifi's other proposed rate
remains active, so proposed selection is not yet final. Baseline and legacy
alternatives are in their final epoch. All22 successful preparation/smoke
bundles and the preserved failed pre-step local attempt validate locally.
Terminal launcher logs, including the planned Riri/Nom handoff stops, are
also retained locally under `logs/transport/`.

At02:49UTC, the remaining baseline and legacy finalists are collected and
validated, including their final solver gates. `baseline_c0_d1_e10` reaches
74.62%/CE0.722067; center75.26% wins the frozen accuracy-first comparison.
`legacy_c2_d0_e10` reaches74.12%/CE0.743407; center74.68%/CE0.717528 wins both
accuracy and CE. Both schemes therefore retain their center rates. All34
completed scientific cases are locally validated. Loulou/Trex/Riri parents
and workers have exited and their GPUs are released. Only `ours_c2_d0_e10`
on Fifi remains; it exactly matches its original epoch5 accuracy65.84% and
CE0.989811 and is training epoch6. A clearly labeled partial report records
the settled comparisons, full LR vectors, fixed contract and open proposed
selection. Expected Fifi completion remains approximately03:50UTC.

Final reconciliation on September 22, 2026: all 35 scientific cases are
complete, collected and validated locally. Fifi's last proposed candidate
finishes at 75.02%/CE0.693852, with native exit0 and a passing final solver
audit. Select `ours_edge_head_down_e10` at 75.36%/CE0.691127, retaining
`baseline_c1_d1_e10` at 75.26% and `legacy_c1_d1_e10` at 74.68%.
No planned scientific case is missing; only the two triggered edge probes
were required. All 58 canonical bundles validate, including the preserved
failed pre-step attempt. All 35 initial model tensor hashes match, source
snapshot hashes verify, and all six ten-epoch runs exactly reproduce their
source run's first five epochs. Every scientific final solver gate passes.

All six production hosts have released their study workers. Native successful
scientific time totals 54.4460 worker GPUh; the failed startup plus reported
preparation/smoke time gives 54.6697 recorded worker GPUh, within the 72h cap.
Concurrent Trex workers are counted separately. The complete local evidence,
named failure/replacement, superseded transports and worker-release checks
are reconciled in `closeout_validation.json` and `worker_release_checks.json`.
The final report contains all raw conv/head/BN/gain rates and comparison
plots. `docs/experimental_manifest.md` records the reviewed interpretation:
use these short-horizon settings for subsequent analog L8 comparisons,
retaining trainable BN, augmentation and cross-entropy. One seed, the
lowest-tested proposed head LR and the short horizon limit generalization;
the proposed 0.10pp lead over baseline does not establish an accuracy
advantage. No 50-epoch run or official-test evaluation is launched.
