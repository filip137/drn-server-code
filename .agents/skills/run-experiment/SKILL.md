---
name: run-experiment
description: Compatibility entry point for running, resuming, monitoring, recovering or collecting a concrete assigned experiment. Routes to the Codifier or Monitor role in experiment-loop.
---

# Run an assigned experiment

Use [experiment-loop](../experiment-loop/SKILL.md), which owns all role handoffs.

- Implement, smoke, launch or recover: enter **Codifier** with the campaign and
  experiment paths, or the existing non-campaign plan and exact configs.
- Monitor, report status or collect: enter **Monitor** with the execution handoff.
- Complete a full assigned experiment: continue through **Reviewer** after collection
  and validation. A status-only request does not authorize new launches.

Read the role-specific context named there. Follow
[execution policy](../../../docs/experiment_workflow.md) for resources and operations
and [reporting](../../../docs/experiment_reporting.md) for canonical evidence.
Do not recreate planning, require a campaign migration, or add an approval gate for
already authorized work. This entry point adds no separate execution policy.
