---
id: "exp-002"
title: "Import current p99 training and checkpoint-specific diagnostics"
status: "imported"
hypotheses: ["H-002", "H-003", "H-004"]
---

# exp-002 — Current training and replay evidence import

## Scope
Import the separately labelled study families in [beta_study](../../../../../docs/beta_study.md):
p99 legacy/ours 30-epoch comparison; p99 ours T8/T16 ten-epoch repeat; p99 checkpoint
replay; initialization beta/T/K replay; and older p90-trained checkpoint replays.
This grouping indexes a synthesis; it is not one matched experimental surface.

## Controls and limitations
Training beta and replay beta are different fields. T affects free settling; K changes
the nudged phase and, in older studies, also the BPTT reference. Separate old/new
weights and environments. Source seed0 continuations are not independent replications.
No unit-RMS-policy training comparison is reported, so H-002 remains not tested.

## Outputs and budget
Keep all original result paths in beta_study.md. Import only, zero compute.
See [result](../results/exp-002-import.md); exp-003 audits the authoritative local files.
