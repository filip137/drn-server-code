# Compute forecast for the matched baseline revision

Updated 2026-09-14T09:09:09.618950+00:00. **Approximately 338.4 GPU-hours** for the campaign, excluding future checks, retries and official-test evaluation. The approved cap remains **300 GPU-hours**; the proposed375-hour cap and faster parallel placement are unapproved.

Settled cost is 228.841717 GPU-hours. Unsettled waves are counted from launch, including their already-collected cases; this is a cost forecast, not a missing-run count. Current training coverage is195/216, with21remaining.

| Work | Cases counted for cost | Projected GPU-hours | Basis |
|---|---:|---:|---|
| Loulou existing three-case original EqProp wave | 3 | 12.44 | Whole unsettled wave; measured first case about4.15h. Two cases are already collected. |
| Nom existing two-case original EqProp wave | 2 | 12.39 | Whole unsettled wave; first case about6.19h and already collected. |
| Remaining Main revised Conv3 BPTT repeats | 4 | 9.20 | Current case plus seed2 across three ceilings;2.3h/case, recent completed case2.2725h. |
| All revised Conv2 EqProp repetitions | 6 | 13.38 | Current Akib case plus five unstarted; mean of three complete pilots2.230463h/case. |
| All revised Conv3 EqProp trainings | 9 | 62.13 | UNMEASURED estimate from native Conv3 EP cost times (T+2K) ratio40/24; includes3pilots and6repetitions. |

The largest uncertainty remains the unmeasured revised Conv3 EqProp runtime. Current estimates use serial lab placement and no sharing speedup. Jean Zay outsourcing is authorized with a four-GPU total concurrency limit, but current access is unavailable during its scheduled maintenance. No acceleration or saving from a future Jean Zay allocation is assumed.

The separate faster parallel plan forecasts about350GPU-hours and requests a375-hour cap to allow measured slower3090 cases and margin. Admissions remain within300; the forecast is not authorization. Scientific case caps and numerical/full-pilot gates remain unchanged.

[Previous337.2-hour forecast](compute_projection_20260914T0354Z.md) · [Initial330.5-hour forecast](compute_projection_initial_20260913.md) · [Approved launch plan](../../docs/paper_training_completion_launch_plan_20260911.md) · [Revision tracker](README.md)
