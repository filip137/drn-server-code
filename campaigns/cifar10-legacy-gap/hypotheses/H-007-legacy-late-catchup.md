---
id: "H-007"
title: "Legacy closes both accuracy deficits during the bounded extension"
---
# H-007 — Legacy catches up with more epochs

## Provenance and claim

Proposed from [X-003](../explorations/X-003-training-horizon.md), before new
measurements. In the unchanged-schedule seed 0 extension, raw legacy's mean
accuracy differences to both baseline and ours improve by at least 0.20 pp
between epochs 46–50 and 66–70. This is partial catch-up, not necessarily a
ranking reversal or proof of a universal convergence-speed difference.

## Decision

Define D_b(W)=accuracy_legacy(W)-accuracy_baseline(W), and likewise D_o.
Support if both D_b(new)-D_b(old) and D_o(new)-D_o(old) are at least +0.20 pp.
Contradict the bounded conjunctive claim if all required arms qualify and
either difference is below +0.20 pp; describe mixed outcomes explicitly.
Invalid/missing arms are inconclusive. Report actual endpoint ordering and
normalized legacy separately; it cannot substitute for raw legacy.

The 0.20 pp threshold is a practical design choice, not a significance test.
No catch-up by 70 does not rule out later catch-up or another schedule.
[exp-006](../series/003-training-horizon/experiments/exp-006-floor-continuation.md)
shares the four continuations with H-006, without duplicate runs.
