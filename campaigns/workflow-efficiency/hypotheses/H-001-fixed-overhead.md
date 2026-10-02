---
id: "H-001"
title: "Fixed workflow overhead exceeds tiny-run execution time"
---
# H-001 — Fixed overhead

Prospective definition for exp-002, amended before execution: from the scientific design-frozen event to the start of scientific review, elapsed time outside smoke and training subprocesses exceeds their combined wall time. Support if overhead > execution; contradict otherwise; inconclusive if timing is incomplete. The original unexecuted exp-001 and user-discussion waits are reported separately, not counted as agent inefficiency. This is a practical engineering threshold, not a statistical test.

Use interval unions: Monitor preparation can overlap execution and must not be double-counted. CPU process wall time includes imports/data loading/reporting; epoch times give a narrower training/evaluation measure. Tokens/inference-only time are not measured. One tiny local pilot does not characterize long-run or remote costs. Campaign creation and audit instrumentation are disclosed overhead.
