---
id: "exp-001"
title: "Import the completed unit-output-RMS calibration"
status: "imported"
hypotheses: ["H-001"]
---

# exp-001 — Unit-output-RMS calibration import

## Scope and source
Historical direct replay, September 22, 2026. Source:
[unit-RMS note](../../../../../docs/conv3_unit_output_displacement_beta_20260922.md).
Three schemes, saved seed-0 initializer, 36 batches of 16, clean endpoints, T=K=8.
Includes both signs and centered displacement. No training or accuracy selection.

## Existing outputs and limits
The source reports three successful replays and three successful smokes; an earlier
OOM smoke was retained and retried with allocator/workspace changes. The original
output root and config are linked in the source. No output was copied or revalidated
in this setup. The numerical tolerance in H-001 is retrospective, not preregistered.

## Budget / result
Historical import only: zero compute. See [result](../results/exp-001-import.md).
