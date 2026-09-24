# CIFAR L8 legacy: focused ten-epoch learning-rate search

Filip requests a learning-rate search for legacy amplification on Fifi,
evaluating each setting for ten epochs. Treat this as fresh optimization
selection from the common saved initializer, not continuation from epoch30.
One exploratory seed0; only the existing45k/5k CIFAR training/validation split.
No official test reads. The previous factor-three coarse grid favored the
legacy center; its two ten-epoch finalists scored74.68% (center) and74.12%
(conv3,head1/3) on Riri. Refine within that bracket using factor-two changes.

Legacy is not consistently worse at matched epochs: at20 legacy82.06% versus
proposed83.10%; at26 legacy85.94% versus proposed85.52%. These continuations
use different GPU/software stacks and do not establish a scheme ranking.
Keep all new LR candidates and a fresh center control together on Fifi.

| Candidate | Conv multiplier | Classifier multiplier | Classifier initial LR |
|---|---:|---:|---:|
| legacy_center |1|1|0.00016|
| legacy_conv_half |0.5|1|0.00016|
| legacy_conv_double |2|1|0.00016|
| legacy_head_half |1|0.5|0.00008|
| legacy_head_double |1|2|0.00032|

The center convolution vector is [0.0045,0.00069,0.00069,0.00069,0.00049,
0.00049,0.00049,0.000345], retaining ratios when multiplied. BN LR0.001 and
gain LR0.00005 remain fixed. Preserve the complete legacy L8 contract:
voltage/current amplification4/0.25, perfect diode, blocks[3,3,2], logical
widths128/256/512, analog dense head, Adam, batch32, trainable affine BN and
gains, crop/flip, CE, bounds[1e-7,10], seed0, matched absolute epoch ordering,
and original50epoch cosine horizon with final ratio0.02. No rho search.

Run all five candidates for ten full epochs; do not prune by intermediate
accuracy. Rank eligible cases by final epoch10 validation accuracy, then
lower final CE, then lexicographically smaller complete conductance LR vector.
Eligibility requires finite full coverage (45000/5000 per epoch),14070 Adam
steps, native exit0, a validated canonical bundle and passing final solver
audit. The runner's best-epoch checkpoint is saved but does not rank settings.
Selection is among these five local probes, not a global LR optimum. This
one-factor-at-a-time search does not resolve conv/classifier interactions;
report a boundary winner without silently expanding the search.

Use the already tested GPU-only CIFAR runner snapshot and accepted legacy
preparation from `cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1`.
Same initializer, split and qualification hashes for all cases. T/K[6,6,4]
is inherited; final solver audit repeated in every candidate. Exact config
two-batch smokes on Fifi must pass before production. The local GPU is occupied
by the legacy continuation, so use destination smokes without disturbing it.
No CPU fallback/offloading. Fresh Adam/scheduler per candidate.

Configs: `configs/cifar/legacy_lr_refinement_20260922/`.
Result root: `results/cifar10-l8-legacy-lr-refinement-e10-seed0-20260922-v1/`,
remote prefix `/home/filip/server_code/`. Target Fifi RTX5090 is idle; all
inventory hosts and Jean Zay checked before allocation. Local/Nom/Trex/Riri/
Loulou are occupied; Akib idle; Jean Zay running an unrelated production job.
No other target is allocated by this search.

Original plan: five sequential runs, about2.1h each,10.5h expected total,
production cap15h. This is superseded by Filip’s02:00Paris September23 hard deadline
below; the five-case search may be incomplete. Per-case3h guards remain,
but the earlier wall deadline always wins. Direct
detached shell loop records command, per-case PID/log/exit and launcher state.
Stop the lane on an operational failure; diagnose before retrying within the
remaining budget. A finite but low-accuracy result is not a failure. Retain
all failed outputs. A completed case failing its final audit is ineligible.

Check launcher, GPU, worker, heartbeat, metrics and logs after launch twice
and at least every30min. Expect progress about every50–60s and an epoch about
13min;45min without progress triggers diagnosis. Collect completed bundles
locally and validate all five before final selection. Maintain live status
and coverage in current_simulations; curate the conclusion in the manifest
after collection and review. No additional scientific cases are authorized
by this bounded plan without revising its visible budget and rationale.

## Superseded deadline: September22 at22:00Paris

Filip subsequently requires a hard stop by22:00CEST today (20:00UTC).
This overrides the original15hour lane budget. Production started15:07UTC,
so at most about4h53min of lane time is now available. Expect roughly two
complete10epoch candidates; any later case still running is stopped and
excluded from the10epoch ranking. Unstarted cases remain explicitly missing.
Do not resume or retry after this deadline without a new user instruction.

An independent detached guard on Fifi selects only this result root’s launcher,
timeout/trainer and data-loader processes, verifying process start identities
before signalling. Its inspect mode confirms exactly those intended processes.
The measured Fifi clock is31.1seconds behind the reference clock; the guard
compensates for this and waits using a monotonic deadline. Termination begins
30seconds before the user deadline, with forced termination10seconds later,
leaving20seconds margin. The launcher is frozen first to prevent queue advance.
Completed results and saved epoch checkpoints remain intact; incomplete runs
receive a user-deadline failure status and exit143. The monitor may collect
artifacts after22:00, but no scientific compute may continue past the cutoff.
Guard/config/status live in hard_stop.py, hard_stop.json and logs/hard_stop.*.

Guard PID2449353 verified armed on Fifi. Launcher2448005 and center worker
2448024 continue normally until the cutoff. Passive collector796740 remains
active; all five smoke bundles are now collected and validate locally.

## Active hard deadline: September23 at02:00Paris

Filip extends the cutoff to02:00CEST September23 (00:00UTC), superseding22:00.
The total lane window is now about8h53min from its15:07UTC launch, enough for
roughly four complete10epoch evaluations at observed speed. Any fifth case
still running is stopped and excluded from the ten-epoch ranking. Existing
case order, scientific configs and training are unchanged; no retry after the
new deadline without further user instruction.

Old guard2449353 was stopped and replaced by guard2450904; the replacement
is verified armed for2026-09-23T02:00:00+02:00. Clock compensation and the
30second termination /10second grace remain. Old22:00 contract/status are
preserved as logs/hard_stop.previous_2200*.json. Training was uninterrupted.

## User-authorized Riri parallel lane

Filip reports Riri free and asks to use it. Live query confirms an idle
RTX5090 (2MiB,0%) and matching PyTorch2.11.0+cu128/CUDA12.8/cuDNN91900.
Move the two unstarted classifier cases (half/double) to Riri; Fifi retains
center and convolution half/double. No added scientific candidates or changed
rates, cohorts, initialization, schedule, batch size or selection rule.
The user-authorized host split supersedes the initial Fifi-only placement.
Host remains recorded; matching software/hardware is not a bitwise guarantee.

The previous source snapshot is preserved as source-before-riri. Only the
transport config lists and their source-identity entries change. Fifi's
already-open list is shortened in place after its unchanged three-case prefix;
its launcher is briefly paused for that update, but the GPU trainer continues.
Current center manifest/source identity stays unchanged and resolves to the
retained prior snapshot. New launches record the revised transport identity.

Riri runs exact-config two-batch smokes before its two10epoch cases, with the
same prepared assets verified by hashes. Expected4.2h on Riri, around22:30CEST;
Fifi's three cases should finish around23:30CEST. Riri cap6h, per-case3h;
combined maximum remains15GPUh but the02:00CEST Sep23 wall cutoff overrides
both lane budgets. Riri's wall clock is roughly two hours slow; use measured
clock-offset compensation and a monotonic deadline guard on that host too.
Result root remains the same study path under each host's server_code/results;
case directories are disjoint and collection reconciles all five cases.

Riri launched after both exact-config smoke bundles passed and their initializer,
split and qualification hashes matched the inherited assets. Launcher992947,
GPU worker992960, deadline guard992883 armed for02:00CEST. First semantic
artifact confirms legacy_head_half epoch1/10, step1. Riri clock offset
-7221.50s is accounted for in both the stop guard and heartbeat-age checks.
Fifi continues center at epoch6/10; no scientific job was interrupted.
Dual-host passive collector847089 replaces796740, verifies both worker/GPU
states, collects disjoint case bundles and waits for both lane exit receipts
before reconciling five-case coverage. It records alerts but does not retry.

## Terminal review, September23

All five cases completed before02:00CEST; both lane exits0. Fifi finished
about23:27CEST; Riri about23:18CEST after clock-offset correction. Five full
production bundles and seven smokes collected and locally validated, full
10epoch coverage and14070 Adam steps each, all solver audits pass. No failed,
excluded, missing or deadline-stopped case. No official test access.
Current rates remain best:74.68%/CE0.717528; head half74.44%, head double74.34%,
conv half74.26%, conv double70.80%. Production11.3943GPUh. Retain current
rates; no further run launched. Report, comparison CSV, selected config and
verification under analysis; scientific interpretation recorded in manifest.
