# Compute forecast for the matched baseline revision

Updated 2026-09-13T22:33:27.192714+00:00. **Approximately 330.5 GPU-hours** for the full campaign under the revised T/K, before remaining checks or retries. The approved cap remains **300 GPU-hours**; this forecast does not authorize additional work.

Settled cost is 182.225345 GPU-hours. The projection counts each unfinished wave from its launch and all 36 revised baseline trainings, so elapsed but unsettled GPU time is not omitted or counted twice.

| Work | Runs | Projected GPU-hours | Basis |
|---|---:|---:|---|
| Existing Loulou Conv3 ours seed-1 wave | 3 | 12.37 | Measured first-case epoch rate; whole wave from launch |
| Existing Nom Conv3 legacy seed-1 wave | 2 | 12.41 | Measured first-case epoch rate; whole wave from launch |
| Existing Jean Zay Conv3 ours seed-2 array | 3 | 13.01 | Measured epoch rate for each allocated task; whole array |
| All revised Conv2 baseline BPTT | 9 | 7.32 | Measured new Akib epoch rate |
| All revised Conv3 baseline BPTT | 9 | 19.57 | Measured new local epoch rate |
| All revised Conv2 baseline EqProp | 9 | 21.45 | Unmeasured forecast: prior full Conv2 EqProp cost times (T+2K) ratio 24/18 |
| All revised Conv3 baseline EqProp | 9 | 62.13 | Unmeasured forecast: prior full Conv3 EqProp cost times (T+2K) ratio 40/24 |

The revised BPTT and existing EqProp wave estimates use observed epoch rates. Revised EqProp durations are still unmeasured: the simple iteration-count scaling assumes comparable time per relaxation step. Target differences, kernel overhead and future sharing may change actual cost. Replace these estimates with measured pilot timings as soon as available.

A proposed 350 GPU-hour cap would add 50 hours to the campaign limit and roughly 20 hours above this central forecast. It would keep the same 36 revised baseline cases, three seeds, eight-hour case limits, preferred GPUs, and one-weekday-5090 restriction. Additional budget is conditional on the numerical and full-pilot gates; a failed scientific group remains held.

The current 295.225345-hour commitment is a conservative admission reservation, not this runtime forecast. Existing admitted work remains within 300 hours. Future admissions above the approved cap require Filip to raise it.

[Approved launch plan](../../docs/paper_training_completion_launch_plan_20260911.md) · [Revised training tracker](README.md)
