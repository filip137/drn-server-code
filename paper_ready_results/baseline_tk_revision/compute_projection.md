# Compute forecast for the matched baseline revision

Updated September 14 at 14:26 CEST. New admissions are paused for Filip’s review. **Approximately 335.6 GPU-hours** for the complete campaign under the current placement. This excludes future checks, retries and official-test evaluation. The authorized cap remains **300 GPU-hours**; the proposed 375-hour cap and faster placement remain unapproved.

The current contract has **201/216 collected results, with 15 training runs remaining**. Settled cost is 262.418304 GPU-hours. The forecast below counts each running case from its start, together with unstarted cases, so already-consumed time in active reservations is included exactly once.

| Remaining work | Cases | Projected GPU-hours | Basis |
|---|---:|---:|---|
| Conv3 baseline BPTT repetitions | 3 | 6.90 | Three unstarted; about 2.3 hours per completed repeat. |
| Conv2 baseline EqProp repetitions | 3 | 6.72 | Three unstarted; about 2.24 hours per completed pilot/repeat. |
| Conv3 baseline EqProp pilots and repetitions | 9 | 59.56 | One running and eight unstarted; first two Loulou epochs project 6.618 hours per case. This remains a preliminary estimate. |

The largest uncertainty is the revised Conv3 EqProp runtime across complete cases. Its first two epochs average 13.24 minutes, consistent with the current eight-hour limit. The Nom diagnostic independently projects 9.788 hours per case, or 11.256 hours with a 15% margin; it cannot run under the current eight-hour limit.

Moving four Conv3 EqProp cases to Nom under the pending parallel plan projects approximately **348.3 GPU-hours** in total. The proposed 375-hour cap provides margin. The measured two-worker Loulou benchmark gives no throughput benefit, so no packing speedup is assumed.

The full outstanding reservations remain 9 GPU-hours: **271.418304/300 committed**. No forecast changes the budget or admits a job. Jean Zay is authorized for at most four concurrent campaign GPUs; its current outage and recorded maintenance prevent assigning a reliable speedup.

[Previous 338.4-hour forecast](compute_projection_20260914T0909Z.md) · [Initial forecast](compute_projection_initial_20260913.md) · [Parallel proposal](../../docs/paper_training_parallel_acceleration_proposal_20260914.md) · [Approved launch plan](../../docs/paper_training_completion_launch_plan_20260911.md)
