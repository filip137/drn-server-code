# Double-Diode 3H/128 Component Timing

- Run: `/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_128/20260313-155212_double_diode_exponential/codex_cpu_component_timing/20260320-151447_double_diode_exponential`
- Validation wall time: `21.953 s`
- Validation accuracy: `96.11%`
- State file is bit-identical to the earlier CPU rerun, so the same low SPICE error applies: `p90=1.187989e-05`, `p99=1.316274e-05`.

## Core Iteration Counts

- Outer iterations: `avg=16.175` over `360` samples
- Newton iterations: `total=119966`, `calls=17469`, `avg_iters_per_call=6.867`
- Estimated total outer sweeps: `5823`
- Expected nonlinear updater calls: `17469`
- Expected all updater calls: `23292`

## Timing Breakdown

- `b` coefficient evaluation: `1.953 s` (8.9% of wall)
- `a` coefficient evaluation: `0.764 s` (3.5% of wall)
- Lambert solver total: `17.285 s` (78.7% of wall)
- Lambert-W substep: `3.171 s` (14.4% of wall)
- Newton polish: `12.446 s` (56.7% of wall)
- Lambert-other work (assembly/masks/exponentials minus Lambert-W and Newton): `1.668 s` (7.6% of wall)
- Other updater overhead outside `a/b/lambert`: `0.401 s` (1.8% of wall)
- Outside-updater overhead for the full validation loop: `1.549 s` (7.1% of wall)

## Per-Call Averages

- All updater calls: `23292`
- Per updater call: `total_pre_activate=0.876 ms`, `b=0.084 ms`, `a=0.033 ms`
- Per nonlinear updater call: `lambert=0.989 ms`, `lambert_w=0.182 ms`, `newton=0.712 ms`, `lambert_other=0.096 ms`

## Interpretation

- This is a clean case for inner-vs-outer analysis because reject/backtrack is disabled.
- Most of the runtime is inside the nonlinear updater itself: `total_pre_activate` is about 93% of wall time.
- Within the nonlinear solve, Newton polish is the dominant cost. It is roughly four times the Lambert-W time here.
- The run is not slow because of inaccurate global convergence alone: outer iterations are moderate (`16.175`). It is expensive because each nonlinear updater call spends substantial time in Newton polish.
- The timing counters are consistent with the architecture: about `5823` outer sweeps, `3` nonlinear hidden layers, and thus about `17469` nonlinear updater calls.
- Because the state file is bit-identical to the previously accepted CPU rerun, this timing breakdown describes the same low-error solution already used for analysis.

## Files

- `timing_breakdown.json`: machine-readable summary
- `component_timing.csv`: flat component table
- `component_timing_bar.png`: simple bar chart
