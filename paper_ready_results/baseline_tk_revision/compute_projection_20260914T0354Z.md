# Compute forecast for the matched baseline revision

Updated 2026-09-14T03:54:27.560409+00:00. **Approximately 337.2 GPU-hours** for the campaign, excluding future checks, retries and official-test evaluation. The approved cap remains **300 GPU-hours**. The proposed 350-hour cap is still unapproved.

Settled cost is 206.512845 GPU-hours. Each unfinished wave below is counted from its launch, including completed but unsettled cases. Settled work is not counted twice.

| Work | Runs | Projected GPU-hours | Basis |
|---|---:|---:|---|
| Loulou existing three-case original EqProp wave | 3 | 12.44 | Whole wave; measured first case about 4.15 hours |
| Nom existing two-case original EqProp wave | 2 | 12.39 | Whole wave; measured first case about 6.19 hours |
| Main revised Conv3 BPTT seed-0 references | 3 | 6.40 | Whole wave; first two cases measured about 2.13 hours each |
| Jean Zay revised Conv2 BPTT seed-2 array | 3 | 6.69 | Measured seed-1 three-task array cost |
| Akib current Conv2 EqProp middle pilot | 1 | 2.23 | Measured first revised pilot cost |
| Remaining Main Conv3 BPTT seeds 1/2 | 6 | 12.80 | Measured reference rate, serial placement |
| Other revised Conv2 EqProp trainings | 7 | 15.57 | Seven cases times first full revised pilot cost 2.225 hours |
| All revised Conv3 EqProp trainings | 9 | 62.13 | UNMEASURED forecast: prior Conv3 EqProp cost times (T+2K) ratio 40/24 |

The first revised Conv2 EqProp pilot now supplies a measured runtime. Conv2 BPTT repetitions use three concurrent V100 allocations because the preferred lab GPUs are occupied; their measured cost is higher than the original all-Akib forecast. The revised Conv3 EqProp cost is still unmeasured and is the largest uncertainty. This forecast assumes current serial lab placement and no sharing speedup.

[Initial 330.5-hour forecast](compute_projection_initial_20260913.md) is retained. A 350-hour cap would leave roughly 13 hours above the new central estimate. Current admissions remain within 300; a forecast or pending proposal does not authorize exceeding it. Scientific case maxima and numerical/pilot gates remain unchanged.

[Approved launch plan](../../docs/paper_training_completion_launch_plan_20260911.md) · [Revision tracker](README.md)
