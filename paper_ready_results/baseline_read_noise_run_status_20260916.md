# Baseline EqProp read-noise training progress

Updated 2026-09-17T22:43:40.711725+00:00. **22/22 new trainings collected and validated.** The declared scope is 20 noisy runs plus two new clean controls, seed 0 only.

T=K=4/6/8. Conv1 beta100/200; Conv2 beta100; Conv3 beta10. Conv3 beta10 remains unconfirmed across seeds and retains its equilibrium caveat. These are validation diagnostics; no official-test evaluation.

| Model | Beta | Sigma | Host | State | Epochs | Best val. % | Final val. % | Final clean loss, pp |
|---|---:|---:|---|---|---:|---:|---:|---:|
| conv1 | 100 | 1e-05 | akibscomputer | complete_collected_validated | 10/10 | 96.18 | 96.04 | 0.00 |
| conv1 | 100 | 3e-05 | akibscomputer | complete_collected_validated | 10/10 | 96.18 | 96.04 | 0.00 |
| conv1 | 100 | 0.0001 | akibscomputer | complete_collected_validated | 10/10 | 96.20 | 96.00 | 0.04 |
| conv1 | 100 | 0.0003 | akibscomputer | complete_collected_validated | 10/10 | 96.18 | 96.02 | 0.02 |
| conv1 | 100 | 0.0005 | akibscomputer | complete_collected_validated | 10/10 | 96.18 | 96.04 | 0.00 |
| conv1 | 200 | 0 | nom-cool-1 | complete_collected_validated | 10/10 | 96.16 | 96.02 | 0.00 |
| conv1 | 200 | 1e-05 | nom-cool-1 | complete_collected_validated | 10/10 | 96.16 | 96.02 | 0.00 |
| conv1 | 200 | 3e-05 | nom-cool-1 | complete_collected_validated | 10/10 | 96.16 | 96.02 | 0.00 |
| conv1 | 200 | 0.0001 | nom-cool-1 | complete_collected_validated | 10/10 | 96.16 | 96.02 | 0.00 |
| conv1 | 200 | 0.0003 | nom-cool-1 | complete_collected_validated | 10/10 | 96.18 | 96.02 | 0.00 |
| conv1 | 200 | 0.0005 | nom-cool-1 | complete_collected_validated | 10/10 | 96.18 | 96.04 | -0.02 |
| conv2 | 100 | 1e-05 | local | complete_collected_validated | 30/30 | 97.30 | 97.30 | -0.08 |
| conv2 | 100 | 3e-05 | local | complete_collected_validated | 30/30 | 97.38 | 97.38 | -0.16 |
| conv2 | 100 | 0.0001 | local | complete_collected_validated | 30/30 | 97.36 | 97.36 | -0.14 |
| conv2 | 100 | 0.0003 | local | complete_collected_validated | 30/30 | 97.34 | 97.34 | -0.12 |
| conv2 | 100 | 0.0005 | local | complete_collected_validated | 30/30 | 97.24 | 97.24 | -0.02 |
| conv3 | 10 | 0 | fifi | complete_collected_validated | 30/30 | 97.64 | 97.64 | 0.00 |
| conv3 | 10 | 1e-05 | fifi | complete_collected_validated | 30/30 | 97.48 | 97.48 | 0.16 |
| conv3 | 10 | 3e-05 | fifi | complete_collected_validated | 30/30 | 97.08 | 97.04 | 0.60 |
| conv3 | 10 | 0.0001 | fifi | complete_collected_validated | 30/30 | 96.70 | 96.64 | 1.00 |
| conv3 | 10 | 0.0003 | fifi | complete_collected_validated | 30/30 | 96.52 | 96.48 | 1.16 |
| conv3 | 10 | 0.0005 | fifi | complete_collected_validated | 30/30 | 96.10 | 96.04 | 1.60 |

Clean losses use matching-beta controls; a positive loss is degradation. Conv1/Conv2 beta100 reuse audited historical controls. Conv1 beta200 and Conv3 beta10 use new controls when complete. GPU/software noise-stream groups remain separate: shared seeds do not imply identical noise across these environments. The Conv1 beta comparison also crosses host/noise-stream groups.

[Launch plan](../docs/eqprop_baseline_read_noise_launch_plan_20260916.md) · [Per-run CSV](baseline_read_noise_run_status_20260916.csv) · [Collection proof](provenance/baseline_read_noise_collection_20260916.json)
