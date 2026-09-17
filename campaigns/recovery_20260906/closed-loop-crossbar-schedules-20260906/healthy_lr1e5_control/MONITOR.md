# Healthy P&V at the matched rate

Status: **complete**. The worker exited successfully after all 30 epochs;
checkpoint replay, physical invariants, reference hashes and all 30 data
streams passed verification. The first three epochs exactly reproduced the
original healthy LR=1e-5 control, including validation readouts and zero writes.
The comparison figure was visually inspected after rendering.

The new healthy selected test KL is **0.0150825996**, versus **0.0234793389**
at LR=3e-6 and **0.0193484630** for corrupt at the matched LR=1e-5. All three
selected epoch 30. Healthy first writes moved from epoch 19 to epoch 6;
total pulses increased from 9,770 to 75,698. These held-apparent results
support the learning-rate explanation for the earlier reversed ordering on
this one array. No convergence or across-array claim follows from this control.

[Report and figure](analysis/report.md) · [Verification](analysis/verification.json)

One exploratory control: healthy crossbar, constant LR=1e-5, 30 complete
epochs, same original HWA + P&V P0, teacher, data, write noise, verify
tolerance 0.04745 and pulse caps as the previous eight-arm comparison.
The preserved corrupt LR=1e-5 result and healthy LR=3e-6 result are named and
hashed in `plan.json`. Validation KL selects checkpoints, with P0 eligible;
test is evaluated only after training and selection.

Use the unchanged, already validated parent runner and CUDA controller.
The earlier CUDA tests cover this rate and writer; no source change requires
repeating the numerical test suite. Verify live CUDA and source hashes before
launch, then require the original P0, data streams and early no-write
trajectory to agree with existing artifacts.

Local RTX 3090 launch: `launch.py`. Supervisor/worker PIDs, exact command,
source plan, log and outputs appear in `launch.json`; `terminal.json` records
the worker's exit and semantic completion. `run/status.json` updates every
100 minibatches; `run/metrics.jsonl` updates each epoch. Expected duration is
several minutes, refined from actual progress. Codex checks artifacts and
process/GPU truth, with a full watchdog check at least every 30 minutes.

Safe retries repair operational faults only and preserve failed attempts.
Do not change LR, epochs, tolerance, pulse budgets, P0 or data in a retry.
Finish after 30 full epochs, successful checkpoint replay, matched-source/data
checks, test readout, and a direct healthy/corrupt KL comparison figure.
