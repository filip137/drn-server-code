---
id: "exp-013"
title: "Refine relative-noise and output-displacement advantage maps for Conv1/2/3"
status: "complete"
hypotheses: ["H-010"]
---
# exp-013 — Relative-noise advantage map and local refinement

## Question and scope

At which nodewise relative read-noise levels and matched initial output D/F
does ours have its largest layerwise cosine advantage over legacy? Also show
its advantage over baseline and the network's weakest-layer alignment.
See [X-009](../../../explorations/X-009-relative-noise-advantage-ridge.md),
[H-010](../../../hypotheses/H-010-relative-noise-advantage-regions.md), and
[exp012](../results/exp-012-relative-endpoint-noise.md).

Filip authorized launch on Akib, nom-cool-1 and local/nom-cool-2 on September 25,
then explicitly instructed: "skip canary, tests. follow the new monitoring
guidelines". This overrides execution smoke/canary and test requirements for
this exploratory run. No test, parity or solver qualification is claimed.
Persistent run-watch performs routine checks without keeping a model agent
awake; bounded agents handle incidents, the stage transition and completion.
No new training, absolute-noise sweep, hidden-D/P
matching intervention, T/K sweep or solver-qualification campaign is included.

## Shared scientific contract

- Conv1/2/3, shared seed-0 initializer within each architecture; baseline
  (1,1), ours (4,1), legacy (4,0.25), using voltage/current amplification.
- Reuse series005 source configs, initializer paths, exact cohort and explicit
  diode dictionaries in `configs/conv/initial_displacement_noise_conv{1,2,3}_20260925.json`.
  Fixed T=K=4/6/8, gains 40/100/360, float64, weights [0,100], zero frozen biases,
  physical 20-node paired-output squared loss and centered frozen-current EqProp.
- Match r = RMS((v_plus-v_minus)/2) / RMS(post-T free output) on the same
  576-example validation cohort, 36 ordered batches of 16. No official test read.
- Noise: `v_tilde = v + eta * abs(v) * xi`, no floor, on every noninput
  positive/negative endpoint after clean relaxation, including the readout.
  Filip explicitly clarified that each sweep point has ONE global eta shared
  by every noisy layer and by all three schemes. No layer-specific eta or
  layer-dependent rescaling of eta is allowed. Absolute noise amplitudes still
  differ with the local voltage magnitude.
  Filip also requires identical eta ranges and sampled values across Conv1,
  Conv2 and Conv3. This applies to both screening and refinement.
  Independent phases/nodes/layers; raw Gaussian draws paired across schemes,
  eta and r within architecture. Inputs, nudging and BPTT remain clean.
- Cache the clean finite-K BPTT reference per architecture/scheme/batch and
  reuse each clean endpoint pair for all eta/draws at that beta. Preserve the
  native per-layer cosine definition: average batch cosines, not cosine of
  an averaged gradient. No gradients or references are borrowed across schemes.

## Stage A: map

Common output targets: **D/F = 1, 2, 3, 4, 5, 6, 8**.
This resolves the interval between the old targets 2 and 6 and brackets them.
Use ONE common grid of 14 global eta values for Conv1, Conv2 and Conv3:

`3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2`.

Each architecture evaluates the complete 7-by-14 grid for all three schemes.
Every eta is applied simultaneously to every noninput layer, including readout,
and every acquisition measures all layers. The common logarithmic grid spans
both early- and late-layer transitions. Conv1/2 relative-noise optima remain
unmeasured. Each point describes one uniformly configured network; do not
combine different layers' optima into a claimed single network operating point.
Eta is dimensionless: 1e-4 is 0.01%, 1e-2 is 1%.

Compute clean comparisons and pooled displacement on all 576 examples. For the
noisy map use only the first 12 fixed batches (192 examples), with three new
base seeds 111000000, 112000000 and 113000000. This preserves the full-cohort
matching coordinate while reducing the cost of the dense acquisition grid.
Never compare the screen's subset mean directly to an older full-cohort mean.

There are 63 physical contexts (3 architectures x 3 schemes x 7 targets),
294 architecture/target/noise cells, 882 scheme/target/noise cells, and 102,060
layer comparisons including full-cohort clean controls and noisy subset rows.
Additional matching corrections are preparation and remain in the compute cap.

Reuse accepted betas where available. For missing r, propose beta by scaling
the nearest recorded beta by r/r_reference, then measure the actual pooled r
in the clean pass. Permit at most two multiplicative corrections to meet 1%
target tolerance. Record requested/achieved r and injected/base beta; mark
unmatched cases explicitly rather than labelling them matched. This is coordinate
matching for the assigned sweep, not a new T/K or broad beta qualification.

## Stage B: refine and independently reassess the peaks

For each of the six convolution layers across the three architectures, select
the Stage A cell with largest mean ours-minus-legacy BPTT cosine. Include
negative maxima if no positive cell exists. Ties use lower eta then lower r.
Selection uses the three-draw mean; retain every draw and all baseline results.

Construct a four-cell patch `{r_star, r_new} x {eta_star, eta_new}`:

1. At an interior r, choose the immediately adjacent r with the larger margin
   at eta_star, then set r_new to their arithmetic midpoint. Ties choose lower r.
   If the maximum is at r=1 or r=8, use r_new=0.5 or 10 respectively.
2. At an interior eta in the common grid, choose the immediate neighbor with
   the larger margin at r_star and use their geometric midpoint. Ties choose
   lower eta. At a grid edge, take one outward half-log-spacing step using the
   closest neighbor. Bound eta to [1e-9, 2e-2]; this gives a new point.
3. Take the union of the six patches' (r, eta) coordinates and deduplicate it.
   Freeze this common set before fresh-noise measurements, then evaluate the
   ENTIRE set on Conv1, Conv2 and Conv3, with all three schemes at every point.
   Refinement therefore also uses identical noise values and ranges across depths.

Use all 576 examples with fresh paired base seeds 121000000, 122000000 and
123000000. At most 24 common (r, eta) coordinates (four per convolution),
72 architecture/target/noise cells, 216 scheme/cells, and 69,984 noisy layer
comparisons before deduplication, plus
clean controls for the unique physical contexts. Recompute comparisons at
the original winning cell on this full cohort; screen and refinement are
separate evidence because both the cohort size and noise draws differ.

This provides finer D/F and eta coordinates and fresh-draw reassessment of
the selected neighborhoods. No recursive expansion: if a peak remains at an
edge, report its direction as unresolved within the bounded sweep.

## Metrics, plots and decision

Primary: per-layer `Delta = mean cosine(ours,BPTT) - mean cosine(legacy,BPTT)`
at identical matched coordinates. Report each scheme's cosine, ours-minus-baseline,
and ours-minus-the-better-of-baseline-and-legacy as companion quantities.
Apply H-010's four-cell, three-fresh-draw rule separately by architecture.
Do not turn a positive gap between two near-zero cosines into good alignment.

Report all convolution layers and readout, clean EqProp/BPTT cosine, noisy/clean
EqProp cosine, relative gradient-error norm, per-layer D/F and D/P, actual read
noise RMS, and attained output displacement. Secondary network statistic:
W = min_layer(mean cosine), including readout; report W_ours-W_legacy and its
screen maximum separately. A W peak is not confirmed unless covered by the
fresh-draw patch. Do not equate individual-layer gains with all-layer dominance.

Deliver matplotlib heatmaps of Delta and each scheme's absolute cosine with
identical logarithmic eta axes and D/F axes across architectures. Mark measured
cells, floor/ceiling behavior and selected patches; label interpolation explicitly. Plot eta/(D/F) as a secondary coordinate to
test whether the apparent optimum is a ridge. Tabulate every layer's sampled
peak and the sampled cells within 95% of its largest positive mean gap; this
95% band is descriptive, not a confidence interval. Keep screen and refinement
tables separate, with per-draw ranges and measured peak coordinates.

## Implementation and execution handoff

Extend `experiments/replay_relative_endpoint_noise.py` through readable configs;
it currently hardcodes Conv3, two targets, two levels and the old reference
cases. Generalize depths, case lists, cohorts and noise seeds without changing
exp012's frozen artifacts. New betas cannot reuse residuals from different
endpoints. Do not require old GPU BPTT bitwise hashes on a new host; compute
native references and label any cross-device equivalence unchecked.

Use the existing launcher and canonical run bundles, not another control plane.
Reuse source diode dictionaries explicitly. Keep GPU-only physical/BPTT/gradient
execution, the proven activation-offload option, finite guards and exact cohort
identity. No full parity, residual or solver sweep is a launch prerequisite.
Record actual checks and any retained exploratory waivers truthfully.

Proposed placement, subject to live memory/process checks at admission:

| Surface, both stages | Host | Planning compute cap |
|---|---|---:|
| Conv1, all schemes | akibscomputer, RTX 3080 | 0.50 GPUh |
| Conv2, all schemes | nom-cool-1, RTX 3090 | 1.00 GPUh |
| Conv3, all schemes | nom-cool-2/local, RTX 3090 | 3.50 GPUh |

Run the three surfaces concurrently. Each depth stays on its recorded host
for matched comparisons; these are intended placements, not verified idle lanes.
Nom-cool-2 and local are one GPU. Leave unrelated jobs untouched. GPU absence
or CPU fallback stops the affected lane. The revised proposed cap is **5 GPUh**,
including preparation, matching, any minimal implementation exercise, both
stages and operational retries. Do not silently reduce batch size or precision.

The cap rises from the earlier 3-GPU-hour proposal because all depths now
evaluate the shared noise grid and the union of refinement patches. Do not
omit levels on a slower architecture to fit the cap; report partial coverage
if the complete common-coordinate comparisons cannot be finished.

Planning estimate: roughly **2–3 hours elapsed** with three usable hosts,
extrapolated from exp012's ~0.425 GPUh for six full-cohort contexts; this is not
a measured ETA for the extended runner. Re-estimate after the first context.
Set a hard wall deadline four hours after the eventual production admission.
If the remaining cap cannot fit the next complete comparison, stop and report
partial coverage; do not enlarge the sweep or the budget automatically.

Expected output root (not created by this proposal):
`results/conv123-relative-noise-advantage-map-20260925-v1/`.
Before creating it, register its planned persistent row in current_simulations.
Codifier must fill exact configs/commands/source identities, lane handles,
measured runtime, collection commands and monitoring ownership here at launch.
Collect remote results locally, reconcile coverage, write one result note,
update the manifest/ledger, and send the measured summary and plot links into
the existing chat using the supported completion callback.

## Live execution handoff — September 25

Started Stage A directly at 14:58:56 Europe/Paris, with canaries, smokes and
tests skipped under Filip's explicit instruction. First cases produce matched
coordinates and finite comparison artifacts. Conv1 runs on Akib (wrapper
580280, CUDA worker 580287); Conv3 runs on local (wrapper 2698147, CUDA worker
2698152, tmux `relative-noise-map-conv3-20260925`). Conv2 was initially staged
pending capacity on nom-cool-1. Filip subsequently explicitly authorized sharing
Ben's GPU across all models, including this RTX 3090. Admission now uses verified
ownership and memory headroom; high utilization alone is not a blocker. Ben's
mumax jobs and MPS settings remain untouched. The updated watcher retains its
incident history and the original deadline/budget.
Conv2 launched at 15:07 Paris (wrapper 3102860, CUDA worker 3102865), with
22,672 MiB free before admission and advancing matching batches observed.

Hard wall deadline: **18:58:56 Europe/Paris September 25**, common to all lanes.
Per-lane caps across both stages remain 1,800 / 3,600 / 12,600 seconds. First
production context timings suggest roughly 10 minutes for the Conv1 screen
and 30 minutes for Conv3, subject to cache reuse and actual progress. Conv2's
ETA must use its observed throughput while sharing with Ben.

Scientific runner: `experiments/replay_relative_noise_advantage.py`; configs:
`configs/conv/relative_noise_advantage_conv{1,2,3}_20260925.json`.
The analysis/selection entry point is `experiments/analyze_relative_noise_advantage.py`.
After all three Stage A lanes are collected, `--stage stage_a --prepare-refinement`
freezes the common coordinate union and creates `inputs/stage_b_convN.json`
under the study root. After Stage B collection, `--stage stage_b` produces
the final plots, comparisons and predeclared neighborhood counts.

Exact commands, paths, source snapshot and budget accounting are in the
[transport handoff](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/transport/launch-handoff.md).
Collection, pending nom1 admission, stage transition and final review/chat
callback are assigned through the [monitor handoff](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/monitor/handoff.md).
Routine monitoring uses the persistent run-watch script with 300-second
checks and bounded incident agents; no model agent remains awake polling.
Original watcher PID **2705081** was armed with `--dispatch`; first observations showed
Akib at 7/21 contexts and local at 2/21 with advancing artifacts and no probe
errors. The nom1 admission incident recorded a capacity wait. Monitoring has
one agent at a time, six calls per rolling day and two attempts per incident.
After the sharing authorization, the same watch restarted as PID **2717721**,
retaining invocation/incident history. Its first combined observation showed
all three lanes advancing (Akib 20/21, local 5/21, nom1 matching batch 6), with
no probe/log errors. Only admission policy changed; scientific configs are unchanged.
All outcomes remain exploratory; skipped qualification and cross-device
equivalence are not reported as passed checks.

## Final execution handoff — September 25

Both stages are collected and validated locally; the scoped H-010 review is complete in the [result note](../results/exp-013-relative-noise-advantage-map.md). Stage B finished at 16:01:35 / 16:04:51 / 16:17:35 Paris for Conv1/Akib, Conv2/nom-cool-1, Conv3/local, all exit 0, before the 18:58:56 deadline. Rounded launcher receipts give combined A+B wall times 962 / 1338 / 3216 seconds, within 1800 / 3600 / 12600-second caps; exact Stage A accounting corrections remain preserved. No new jobs or relocations are authorized. The bounded completion agent generated the final report and campaign review. Root-thread callback is unavailable in its tool context; local final summary records the outcome without claiming a delivered notification. Historical central snapshots remain untouched under AGENTS.md.
