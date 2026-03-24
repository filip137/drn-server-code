# Double-Diode 3H/128 Component Timing (Float64)

- Run: `/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_128/20260313-155212_double_diode_exponential/codex_cpu_component_timing_float64/20260320-152054_double_diode_exponential`
- Validation wall time: `19.448 s`
- Validation accuracy: `96.11%`
- SPICE error stays low, but it is not bit-identical to the float32 run: `p90=1.430091e-05`, `p99=1.990818e-05`.
- Direct float64-vs-float32 state difference is small: `p90=1.958773e-06`, `p99=4.522276e-06`.

## Core Iteration Counts

- Outer iterations: `avg=26.5083` over `360` samples
- Newton iterations: `total=39242`, `calls=28629`, `avg_iters_per_call=1.37071`
- Estimated total outer sweeps: `9543`
- Expected nonlinear updater calls: `28629`
- Expected all updater calls: `38172`

## Timing Breakdown

- `b` coefficient evaluation: `3.053 s` (15.7% of wall)
- `a` coefficient evaluation: `1.232 s` (6.3% of wall)
- Lambert solver total: `12.843 s` (66.0% of wall)
- Lambert-W substep: `5.183 s` (26.7% of wall)
- Newton polish: `4.777 s` (24.6% of wall)
- Lambert-other work (assembly/masks/exponentials minus Lambert-W and Newton): `2.883 s` (14.8% of wall)
- Other updater overhead outside `a/b/lambert`: `0.619 s` (3.2% of wall)
- Outside-updater overhead for the full validation loop: `1.701 s` (8.7% of wall)

## Per-Call Averages

- All updater calls: `38172`
- Per updater call: `total_pre_activate=0.465 ms`, `b=0.080 ms`, `a=0.032 ms`
- Per nonlinear updater call: `lambert=0.449 ms`, `lambert_w=0.181 ms`, `newton=0.167 ms`, `lambert_other=0.101 ms`

## Comparison To Float32 Timed Run

- Float64 wall time is lower: `19.448 s` vs `21.953 s` (`11.4%` faster).
- Float64 needs more outer sweeps: `avg outer=26.508` vs `16.175`.
- But it needs far less Newton polishing per call: `avg_iters_per_call=1.371` vs `6.867`.
- The largest timing drop is Newton polish: `4.777 s` vs `12.446 s`.
- Lambert-W time increases in float64: `5.183 s` vs `3.171 s`, but that is more than offset by the smaller Newton cost.

## Interpretation

- This remains a clean case because reject/backtrack is disabled.
- Relative to the float32 timed updater, the float64 updater keeps the hidden Lambert solve and Newton polish in float64 instead of float32 work with only Lambert-W evaluated in float64.
- In this run, the float64 path changes the local solve enough that each nonlinear call needs much less Newton polishing, even though the global equilibrium loop takes more outer iterations.
- The net effect is favorable here: more outer sweeps, but cheaper nonlinear solves and a lower overall wall time.
- The solution is not identical to the float32 one, but it stays in the same low-error regime against SPICE.

## Files

- `timing_breakdown.json`: machine-readable summary
- `component_timing.csv`: flat component table
- `component_timing_bar.png`: float64 component bar chart
- `component_timing_comparison_bar.png`: float32-vs-float64 comparison chart
