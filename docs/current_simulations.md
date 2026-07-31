# Current Simulations

Updated: 2026-07-31

The Active section is generated from local `results/**/status.json` files.
Maintain the other sections manually. Scientific conclusions belong in
[`experimental_manifest.md`](experimental_manifest.md), while terminal
direction belongs in [`current_state.md`](current_state.md).

## Active

<!-- BEGIN GENERATED ACTIVE RUNS -->
_No reporting-contract runs are currently marked `running`._
<!-- END GENERATED ACTIVE RUNS -->

## Recent Experiment Directories

| Results | State | Summary |
|---|---|---|
| [Bounded Conv1/Conv2 checkpoint mechanisms](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/) | `reviewed` | Local RTX 3090 read-only replay of 16 canonical baseline/ours and five completed historical legacy `[1e-5,1e-4]` selections. All 63 checkpoint guards passed on one frozen 256-example cohort. Conv2 baseline Adam is projection-limited; the Kaiming/ours/SGD chance row instead shows dead-gradient collapse. Reviewed conclusions are in `experimental_manifest.md`; two failed smoke attempts and the passing `smoke-attempt-03/` are retained. |

## Queued

None recorded.

## Paused

None recorded.

## Analyzing

None recorded.
