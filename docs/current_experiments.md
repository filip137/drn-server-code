# Current Experiments

## Research-question entry points

New planning for the two active questions lives in [campaigns](../campaigns/README.md):
[read-noise beta](../campaigns/read-noise-beta/README.md) and
[CIFAR-10 legacy gap](../campaigns/cifar10-legacy-gap/README.md).
Their generated ledgers index scientific records; execution handoffs and result paths
live in experiment notes. Small runs use [pilot notes](../campaigns/pilots/README.md).
The old [current_simulations.md](current_simulations.md) is a historical snapshot.
See the [migration map](campaign_workflow_migration.md).

## Preserved historical handoff

The entry below is retained from the pre-campaign snapshot. It is not a fresh
query of live state or authorization to restart the job.

<!-- experiment-id: eqprop-conv3-p99-read-noise-5em4-20260922-v1 -->
### Conv3 p99 read noise 5e-4

- **Testing:** Legacy and ours p99 stability at sigma5e-4, seed0,30epochs; [plan](experiment_plans/eqprop-conv3-p99-read-noise-5em4-20260922-v1.md).
- **Where:** Jean Zay production52725, two V100s; [bundle](../results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/).
- **Status:** `ready-for-review` — Both runs completed 30 epochs; complete local snapshot collected and validated on 2026-09-23. [Summary](../results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/production-02-summary.md). Scheduled updates stopped.
