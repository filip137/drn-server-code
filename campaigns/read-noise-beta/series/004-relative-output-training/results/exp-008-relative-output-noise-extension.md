---
experiment: "exp-008"
evidence: "validated-local"
summary: "108 new trainings completed and validated; nine unstarted Conv3 target-6 cases were withdrawn when the user changed research direction."
verdicts: {}
---
# Coverage record: exp-008

The [assigned extension](../experiments/exp-008-relative-output-noise-extension.md)
is operationally terminal as of September 25, 2026, 11:22:24 Europe/Paris.
All **108 executed new cases completed successfully** and their authoritative
local canonical bundles pass `experiments.reporting.validate_run` with zero
errors. The 117 declared new config identities reconcile exactly to these 108
unique results and nine explicitly withdrawn, unstarted cases; there are no
unexpected results, failed extension attempts, or replacements.

| Architecture | New successful cases | Scope-withdrawn cases | Selected exp-007 reuse | Observed combined cells |
|---|---:|---:|---:|---:|
| Conv1 | 51 | 0 | 12 | 63 |
| Conv2 | 33 | 0 | 3 | 36 |
| Conv3 | 24 | 9 | 3 | 27 |
| Total | 108 | 9 | 18 | 126 |

The nine withdrawals are **Conv3 target 6.0**, all three schemes
(`baseline`, `ours`, `legacy`) crossed with sigma `5e-4`, `1e-3`, and `2e-3`.
The Codifier cancelled pending Slurm tasks `169756_24`–`169756_26` at 11:14:25
Paris, before allocation; scheduler elapsed time is zero. Tasks 27–32 were
never submitted. These are user-directed scope exclusions, not numerical or
operational failures. Active tasks 21–23 were allowed to finish unchanged.
No further training admission or restart is authorized by this record.

The 18 reused controls remain authoritative in exp-007; they are not new
extension trainings. That study's Fifi cancellations and recovered node-paging
attempt remain in its separate provenance. The original 135-cell extension
declaration is retained: 126 observed cells plus nine withdrawals. Scientific
review and hypothesis verdicts are intentionally pending; no reduced-target
mean substitutes for a predeclared full-target mean here.

All new terminal bundles are under
[`results/eqprop-conv123-relative-output-noise-extension-20260924-v1`](../../../../../results/eqprop-conv123-relative-output-noise-extension-20260924-v1/):

- `production/local`: 30 successful cases.
- `collected/akib/production`: 21 successful cases.
- `collected/riri/production`: 18 successful cases.
- `collected/trex/production`: 15 successful cases.
- `collected/jean-zay/production/jz-0` through `jz-23`: 24 successful cases.

Complete canonical artifacts, checkpoints, metrics, source/config identities,
and launch logs/receipts are local. Frozen archive SHA256 is
`5a1e95b8e9e5cf022209f4ea0f8e76c053bdc10ba39ecb8360a5bc736d0cda37`.
The primary analysis remains epoch-10 validation accuracy, not best-epoch
accuracy; official-test evaluation was disabled. Calibration and smokes are
excluded from training evidence.

Final scheduler accounting is preserved in
`collected/jean-zay/slurm-accounting.tsv`: arrays 162366 and 166933 completed
nine cases each, and 169756 completed six cases; all successful exits are 0:0.
At the terminal check no extension Slurm task remained active or queued.
Workstation queues had already completed with exit code zero and released our
workers; unrelated GPU processes were left intact.

Measured production allocation or host-occupancy time totals **58.757 GPU-hours**:
35.434 V100 allocation-hours, 4.251 local, 2.699 Akib, 8.547 Riri, and 7.826 Trex.
These workstation figures use queue start/end intervals, rather than summing
potentially overlapping processes. Riri's host clock was about 7,225 seconds
behind the monitor; its same-host duration remains valid. RTX 5090 runs shared
capacity with authorized unrelated work, so these are occupancy measurements,
not exclusive-device benchmarks. The recorded 117-smoke span (784.806 seconds)
and calibration/smoke measurements (193.083 seconds) add approximately 0.272
GPU-hours, plus small subsequent admission-gate smokes. Observed use remains
comfortably below the separate 100 GPU-hour extension cap; no exact all-overhead
total is claimed.
