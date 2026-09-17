# Open-loop versus closed-loop recovery

This isolated exploratory experiment starts from the exact width-256 HWA +
P&V deployments discussed in the September 4–5 reviews. The main comparison
contains eight arms: DRN/crossbar × healthy/faulted × open-loop/P&V. Each arm
uses three full 55,000-example epochs, batch size 16, the original teacher,
device state, data split and order. `plan.json` freezes the scientific contract.

The intervention is the device writer. Open-loop arms use the original
stochastic single-pulse Adam implementations. Closed-loop arms accumulate
Adam commands in a desired target and issue real SET/RESET pulses until the
held apparent observation reaches tolerance or the pulse budget is exhausted.
The hidden persistent state is used only by the physical plant and explicitly
labelled diagnostics. True bounds and immutable faults are enforced by the
plant. There is no hidden-mask controller shortcut and no noise redraw on a
verify read. This is full pulse-resolved P&V, not a noisy-update approximation.

All arms retain the original cumulative recovery cap of 640 pulses per cell;
P&V additionally allows at most 128 pulses per cell per minibatch. The desired
target accumulates small updates rather than discarding them below tolerance.
Targets outside the reachable device window are retained and reported through
unresolved commands and cap exhaustion; they do not move the physical bounds.

`drn_code/` is a frozen archive of commit `2b283365063931e9fdd41fbd30d2672e0e600f04`.
`crossbar_code/` is the frozen September-5 recovery-canary archive. The full
crossbar revision and all named inputs appear in the plan and run manifests.
Original worktrees and checkpoints remain the source of the inputs.

The DRN has synthetic post-P&V RESET-stuck faults and Figure-6/OM hybrid
devices. The crossbar has native OM effective-weight faults. Teachers and
device models differ, so this tests writer effects within each architecture.
One array/write per architecture supports an exploratory figure, not a
population-level architecture comparison.

The CUDA canaries cover accumulation of sub-tolerance updates, true no-op
behavior, overshoot reversal, stop-on-acceptance, pulse-budget exhaustion,
real SET/RESET writes, exact faulted checkpoint replay, physical bounds,
immutable persistent faults, and unchanged untouched cells.

Two initial canaries are retained separately under `smoke/`: the first DRN
canary used projected feedback before it was corrected to match the original
programmer's unprojected apparent readings; the first crossbar canary failed
selected-state replay because the driver omitted the required fault-source
and healthy-P0 authority arguments. The corrected canaries and physical
checks passed. These artifacts are operational checks, not scientific arms.

Run from this directory with the host CUDA interpreter:

```bash
/home/filip/miniconda3/envs/py312/bin/python launch.py
```

`launch.json` records the parent and expected cases. `logs/*.launch.json`
records every command and PID; `logs/*.exit.json` records exit status and
semantic completion. Each arm writes a heartbeat every 100 minibatches,
epoch metrics, a continuation checkpoint, a validation-selected checkpoint,
and a terminal result. The queue has at most two CUDA workers. A passive
launcher only records state; Codex inspects process, GPU, artifact, and log
progress and diagnoses failures.

Checkpoint selection minimizes held-apparent validation teacher KL with P0
eligible. Both apparent and persistent metrics are reported. After training
and selection, the test set is evaluated at P0, the selected state and the
fixed-final state. The analysis verifies exact paired inputs, states and data
streams, then writes figures and `analysis/report.md`.

The two slow DRN P&V attempts are preserved under `attempts/`. The numerical
plant and controller were fused into CUDA kernels with floating-point fusion
disabled. A 33-round physical check (including empty rounds and pulse caps)
and a 512-minibatch end-to-end replay matched all persistent/apparent tensors,
optimizer tensors and counters, RNG state, data-generator state, and final
metrics bitwise. `run_fast_recovery.py` supplies the operational wrapper;
`acceleration.json` in each DRN P&V result records its source hashes.
`operational_acceleration.json` records the replacement, and
`accelerated_launch.json` pins the six completed results reused unchanged.
