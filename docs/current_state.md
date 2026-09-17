# Current State

Last updated: 2026-08-25

## Purpose

This document is the human-readable synthesis of the repository: what we are
trying to learn, what the evidence currently suggests, and which comparisons
would change our direction. It is deliberately not a run log or a store of
per-study measurements.

- Concluded-study interpretations and artifact locations belong in the
  [experimental manifest](experimental_manifest.md).
- Running, queued, analyzing, and paused work belongs in
  [current simulations](current_simulations.md).
- Exact plans, native runs, checkpoints, and generated analyses belong under
  `results/<study-id>/` as described by the
  [experiment workflow](experiment_workflow.md).

## Central research question

Under which measured-device conditions does hardware-aware training (HWA)
underperform, and when does the remaining deployment gap require an additional
on-chip update rule such as Tiki-Taka or LoRA?

The important comparison is not “does on-chip training improve one run?” It is
whether a predeclared on-chip arm improves over a matched HWA-only deployment
when the logical initialization, DRN topology, solver, data split, physical
device assignment, programmed state, and selection rule are held fixed.

## The experimental logic

We separate the path to deployed performance into five stages:

1. **Logical initialization.** Map a frozen ReLU teacher into an ideal bounded
   perfect-diode DRN and measure any loss before device effects.
2. **Hardware-aware training.** Train from the same logical state using a
   declared device-aware modifier. Compare it with a matched clean-training
   control rather than with an independently trained checkpoint.
3. **Physical mapping and programming.** Assign devices, map logical targets
   into reachable windows, and apply the declared program-and-verify process.
   Keep mapping contraction, unreachable targets, corrupt devices, clipping,
   and programming residuals separately measurable.
4. **Deployment.** Evaluate the exact saved deployed state before recovery.
   Distinguish the apparent forward endpoint from persistent physical update
   state whenever the device model supplies both.
5. **Matched recovery.** Starting from that same deployed state, compare HWA
   only with the declared recovery controls. Candidate controls include soft
   SET/RESET fine-tuning, Tiki-Taka, LoRA, selective rewriting, and full-model
   rewriting.

This decomposition is the organizing principle for new studies. Initialization
quality, HWA quality, mapping loss, programming loss, device reassignment, and
the recovery rule must not be collapsed into one before/after accuracy number.

## Current synthesis

- **The logical starting point is now strong enough to isolate later device
  effects.** Direct zero-update mapping of the frozen ReLU teacher into the
  bounded four-device FP32 DRN was effectively lossless on the tested split.
  This supersedes the independently trained bounded checkpoint as the matched
  initialization control, but says nothing yet about measured-device
  deployability.
- **Physical representation often matters more than nominal weight range.**
  Differential and shared-common-window mappings preserve signed information
  far better than independent one-device affine mappings. A finite floor can
  preserve noiseless ordering while still shrinking voltages and class margins
  enough to make dynamic noise important.
- **Array-specific HWA has shown a narrow benefit, not an adequate deployment.**
  On one fixed counterfactually repaired IBM OM assignment, compact-endpoint
  HWA improved the matched pulse-resolved deployment over clean training, but
  the deployed network remained far below the logical control. The saved
  decomposition attributes meaningful loss to both common-window mapping and
  programming error, with the first layer especially sensitive.
- **Successful per-cell verification is not sufficient network evidence.** A
  deployment can accept nearly every individual cell while differential error
  remains larger than the useful contrast signal. Future gates must include
  network accuracy, teacher agreement, output margin, and layerwise
  differential signal-to-write-noise measurements.
- **Device assignment and reassignment are first-class interventions.** The
  four-device cohort-transfer study showed large variation and severe loss
  across held-out assignments even when aggregate reachable-window statistics
  appeared similar. Population-level claims therefore need fixed,
  predeclared assignments rather than one convenient array.
- **Shared RESET-relative deployment substantially improves transfer, while a
  QAT-specific advantage remains unresolved.** The nine-level QAT checkpoint
  retained 91.00% mean apparent-forward accuracy on the held-out repaired OM
  assignment, but exceeded the matched continuous shared-target HWA arm by
  only 0.71 percentage points; the paired interval crossed zero and the
  predeclared two-point gate failed. The apparent endpoint is the deployed
  forward state, while the persistent endpoint is an analysis-only simulator
  state from which subsequent pulses evolve, not a retention or reread metric.
- **Recovery is possible, but the required intervention is unresolved for the
  current deployment.** Historical controls show that full rewriting,
  sign-only updates, and low-rank adaptation can recover different kinds of
  degradation. The next matched pilot will start direct single-pulse,
  gradient-free refresh, and Tiki-Taka controls from one exact shared
  RESET-relative two-state deployment; until that evidence exists, no specific
  on-chip update rule is established as necessary.
- **Program-and-verify cost has real tradeoffs.** Adaptive pulse batching can
  reduce verify reads while using more pulses and lowering programming
  success. Compact endpoint models must therefore travel with their separate
  failure, corruption, saturation, and programming-cost models.
- **The evidence remains exploratory.** Most conclusions use one teacher or
  base seed, one or a few device assignments, endpoint or fitted device models,
  and incomplete circuit nonidealities. They establish simulation behavior,
  not fabricated-hardware performance.

Exact outcomes, limitations, and local artifact links are indexed in the
[experimental manifest](experimental_manifest.md).

## What would establish a need for on-chip training

An evidence-backed claim requires one study with all of the following:

- a strong, hash-pinned logical initialization;
- a matched clean-training and HWA comparison;
- one fixed device assignment and one saved deployed persistent/apparent state;
- an HWA-only deployment that fails a predeclared performance or margin gate;
- one or more recovery arms that start from that identical deployed state;
- a predeclared recovery target and intervention-cost measurements; and
- replication over independent assignments and, eventually, base-training
  seeds.

If HWA-only meets the deployment gate, on-chip training is not required for
that condition. If HWA-only fails and no matched recovery arm closes the gap,
the evidence identifies an unresolved deployment failure rather than a need
for a particular update rule.

## Current priorities

1. Run one predeclared development seed of matched recovery from the exact
   shared RESET-relative QAT deployment, using the common initialization as
   the no-update reference and comparing gradient-free rail refresh, direct
   one-pulse updates, ideal-fast Tiki-Taka, and physical-fast Tiki-Taka at
   fixed slow-write caps.
2. Use terminal apparent-forward accuracy as the recovery outcome; report the
   analysis-only persistent-state diagnostic and source-gain versus
   recalibrated-gain metrics separately rather than treating either as a
   substitute for deployed inference.
3. If the pilot is viable, add two development endpoint repeats and freeze the
   smallest pulse cap meeting the declared recovery gates before examining
   recovery outcomes on the five saved held-out deployments.
4. Preserve pulse, verify, reversal, failure, saturation, fast-array RESET,
   and unique-cell costs alongside predictive performance.
5. After the pulse-update comparison is frozen, add LoRA and soft-pulse
   controls from the same saved deployment rather than from a newly programmed
   state.

## Open questions

- Which mapping and common-window policy best preserves differential signal
  without hiding passive loading or unreachable targets?
- Does device-matched HWA improve population-level deployment, or only the one
  array used during optimization?
- Which part of the remaining gap is caused by mapping, programming error,
  persistent-state dynamics, read noise, or limited update reachability?
- Can a low-rank branch correct the observed first-layer differential errors,
  or is selective/base rewriting necessary?
- At what accuracy, margin, endurance, and write-cost target does Tiki-Taka or
  LoRA become preferable to off-chip retraining and redeployment?
- Which conclusions survive independent teachers, data splits, assignments,
  and measured incremental-pulse datasets?

## Evidence boundaries

- Device-facing claims require the measured-data and provenance rules in
  [synapse data](synapse_data.md).
- Every study must predeclare its initialization path under the
  [initialization protocols](initialization_protocols.md).
- Endpoint program-and-verify distributions do not by themselves identify
  incremental pulse-update dynamics.
- Apparent acceptance, persistent reachability, and verify-window intersection
  are different facts and remain separately reported.
- Conventional ReLU networks are teachers or references; the evaluated
  student remains a perfect-diode DRN.

## Personal notes

<!-- Add personal notes below this line. -->

## Document map

- [Experimental manifest](experimental_manifest.md) — concise concluded-study
  index, short interpretations, and local result links.
- [Current simulations](current_simulations.md) — active and planned work.
- [Experiment workflow](experiment_workflow.md) — study lifecycle and closeout.
- [Raw result layout](../results/README.md) — study-root directory contract.
- [Initialization protocols](initialization_protocols.md) — matched starting
  states and checkpoint rules.
- [IBM ReRAM program-and-verify model](ibm_reram_program_verify_noise_model.md)
  — device-model and controller contracts.
- [IBM OM HWA/program-and-verify pilot](ibm_om_hwa_program_verify_pilot.md) —
  current device-specific deployment protocol.
