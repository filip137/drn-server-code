# Single-seed EqProp read-noise results and remaining runs

Updated: 2026-09-16T04:22:15.260430+00:00.

**30/30 collected and validated; 0 running; 0 queued/prepared; 0 held.**

Wide Conv1/2/3, ours/legacy, seed 0; no new zero-noise or 1e-3 runs. Ordinary-MNIST validation only; official test disabled. Conv3 ours retains the recorded clean-gradient qualification exception. The first 12 runs finished before the September 15 08:00 deadline. The remaining 18 were authorized on local, Akib, and Nom on September 15; the two queued Conv3 legacy cases at 3e-5/3e-4 moved to Trex/Fifi for the September 16 08:00 deadline.

[Continuation plan](../docs/eqprop_read_noise_continuation_plan_20260915.md) · [Overnight plan](../docs/eqprop_read_noise_overnight_plan_20260914.md) · [CSV](read_noise_run_status_20260914.csv)

Noise-stream audit: local and Nom reproduce the same sampled arrays. The 5090 group and Akib's 3080 form two further, distinct groups. Conv1/2 pairs stay on one host. Conv3 at 1e-5, 1e-4, and 5e-4 compares ours/5090 with legacy/3090 runs; at 3e-5 and 3e-4 it compares ours/3090 or ours/3080 with the deadline-transferred legacy/5090 runs. All five Conv3 pairs therefore differ in GPU/software environment and noise realization. Historical clean references also use a different GPU/software environment.

| Architecture | Sigma | Scheme | Host | State | Best / final validation (%) | Drop from clean, best / final (pp) |
|---|---:|---|---|---|---|---|
| conv1 | 1e-05 | legacy | trex | [complete](bundles/read_noise_20260914/conv1/legacy/sigma_1e-05/seed0/result.json) | 96.52 / 96.40 | -0.04 / 0.04 |
| conv1 | 1e-05 | ours | trex | [complete](bundles/read_noise_20260914/conv1/ours/sigma_1e-05/seed0/result.json) | 96.44 / 96.34 | -0.02 / 0.00 |
| conv1 | 3e-05 | legacy | nom-cool-1 | [complete](bundles/read_noise_20260914/conv1/legacy/sigma_3e-05/seed0/result.json) | 96.50 / 96.40 | -0.02 / 0.04 |
| conv1 | 3e-05 | ours | nom-cool-1 | [complete](bundles/read_noise_20260914/conv1/ours/sigma_3e-05/seed0/result.json) | 96.44 / 96.30 | -0.02 / 0.04 |
| conv1 | 0.0001 | legacy | fifi | [complete](bundles/read_noise_20260914/conv1/legacy/sigma_0.0001/seed0/result.json) | 96.56 / 96.40 | -0.08 / 0.04 |
| conv1 | 0.0001 | ours | fifi | [complete](bundles/read_noise_20260914/conv1/ours/sigma_0.0001/seed0/result.json) | 96.40 / 96.34 | 0.02 / 0.00 |
| conv1 | 0.0003 | legacy | akibscomputer | [complete](bundles/read_noise_20260914/conv1/legacy/sigma_0.0003/seed0/result.json) | 96.52 / 96.36 | -0.04 / 0.08 |
| conv1 | 0.0003 | ours | akibscomputer | [complete](bundles/read_noise_20260914/conv1/ours/sigma_0.0003/seed0/result.json) | 96.44 / 96.30 | -0.02 / 0.04 |
| conv1 | 0.0005 | legacy | loulou | [complete](bundles/read_noise_20260914/conv1/legacy/sigma_0.0005/seed0/result.json) | 96.48 / 96.38 | 0.00 / 0.06 |
| conv1 | 0.0005 | ours | loulou | [complete](bundles/read_noise_20260914/conv1/ours/sigma_0.0005/seed0/result.json) | 96.50 / 96.28 | -0.08 / 0.06 |
| conv2 | 1e-05 | legacy | akibscomputer | [complete](bundles/read_noise_20260914/conv2/legacy/sigma_1e-05/seed0/result.json) | 97.68 / 97.62 | 0.62 / 0.44 |
| conv2 | 1e-05 | ours | akibscomputer | [complete](bundles/read_noise_20260914/conv2/ours/sigma_1e-05/seed0/result.json) | 98.12 / 98.08 | -0.02 / -0.06 |
| conv2 | 3e-05 | legacy | akibscomputer | [complete](bundles/read_noise_20260914/conv2/legacy/sigma_3e-05/seed0/result.json) | 97.40 / 97.38 | 0.90 / 0.68 |
| conv2 | 3e-05 | ours | akibscomputer | [complete](bundles/read_noise_20260914/conv2/ours/sigma_3e-05/seed0/result.json) | 98.08 / 98.08 | 0.02 / -0.06 |
| conv2 | 0.0001 | legacy | nom-cool-1 | [complete](bundles/read_noise_20260914/conv2/legacy/sigma_0.0001/seed0/result.json) | 96.64 / 96.52 | 1.66 / 1.54 |
| conv2 | 0.0001 | ours | nom-cool-1 | [complete](bundles/read_noise_20260914/conv2/ours/sigma_0.0001/seed0/result.json) | 98.02 / 98.00 | 0.08 / 0.02 |
| conv2 | 0.0003 | legacy | local | [complete](bundles/read_noise_20260914/conv2/legacy/sigma_0.0003/seed0/result.json) | 95.98 / 95.94 | 2.32 / 2.12 |
| conv2 | 0.0003 | ours | local | [complete](bundles/read_noise_20260914/conv2/ours/sigma_0.0003/seed0/result.json) | 97.88 / 97.80 | 0.22 / 0.22 |
| conv2 | 0.0005 | legacy | local | [complete](bundles/read_noise_20260914/conv2/legacy/sigma_0.0005/seed0/result.json) | 95.72 / 95.72 | 2.58 / 2.34 |
| conv2 | 0.0005 | ours | local | [complete](bundles/read_noise_20260914/conv2/ours/sigma_0.0005/seed0/result.json) | 97.66 / 97.60 | 0.44 / 0.42 |
| conv3 | 1e-05 | legacy | nom-cool-1 | [complete](bundles/read_noise_20260914/conv3/legacy/sigma_1e-05/seed0/result.json) | 96.12 / 95.68 | 2.72 / 3.10 |
| conv3 | 1e-05 | ours | trex | [complete](bundles/read_noise_20260914/conv3/ours/sigma_1e-05/seed0/result.json) | 98.38 / 98.26 | 0.34 / 0.38 |
| conv3 | 3e-05 | legacy | trex | [complete](bundles/read_noise_20260914/conv3/legacy/sigma_3e-05/seed0/result.json) | 93.42 / 93.42 | 5.42 / 5.36 |
| conv3 | 3e-05 | ours | local | [complete](bundles/read_noise_20260914/conv3/ours/sigma_3e-05/seed0/result.json) | 97.88 / 97.82 | 0.84 / 0.82 |
| conv3 | 0.0001 | legacy | nom-cool-1 | [complete](bundles/read_noise_20260914/conv3/legacy/sigma_0.0001/seed0/result.json) | 84.66 / 83.08 | 14.18 / 15.70 |
| conv3 | 0.0001 | ours | fifi | [complete](bundles/read_noise_20260914/conv3/ours/sigma_0.0001/seed0/result.json) | 97.62 / 97.62 | 1.10 / 1.02 |
| conv3 | 0.0003 | legacy | fifi | [complete](bundles/read_noise_20260914/conv3/legacy/sigma_0.0003/seed0/result.json) | 84.92 / 82.34 | 13.92 / 16.44 |
| conv3 | 0.0003 | ours | akibscomputer | [complete](bundles/read_noise_20260914/conv3/ours/sigma_0.0003/seed0/result.json) | 97.12 / 96.82 | 1.60 / 1.82 |
| conv3 | 0.0005 | legacy | nom-cool-1 | [complete](bundles/read_noise_20260914/conv3/legacy/sigma_0.0005/seed0/result.json) | 76.36 / 75.44 | 22.48 / 23.34 |
| conv3 | 0.0005 | ours | loulou | [complete](bundles/read_noise_20260914/conv3/ours/sigma_0.0005/seed0/result.json) | 96.78 / 96.60 | 1.94 / 2.04 |

Positive drops mean lower accuracy than the matching clean seed-0 reference. Only full, locally validated trainings enter the result table; smoke/timing runs and deadline-truncated runs are excluded. A single seed measures this trajectory, not seed-to-seed uncertainty.
