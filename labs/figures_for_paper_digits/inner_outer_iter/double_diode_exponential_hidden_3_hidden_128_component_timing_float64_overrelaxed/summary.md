# Double-Diode 3H/128 Component Timing (Float64 Overrelaxed)

- Run: `/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_128/20260313-155212_double_diode_exponential/codex_cpu_component_timing_float64_overrelaxed/20260320-153823_double_diode_exponential`
- Validation wall time: `9.794 s`
- Validation accuracy: `96.11%`
- SPICE error remains low: `p90=1.188427e-05`, `p99=1.316690e-05`.
- Direct float64-vs-float32 state difference is tiny: `p90=1.743974e-08`, `p99=2.053855e-08`.

## Iterations

- Outer iterations: `avg=16.1722` over `360` samples
- Newton iterations: `total=24275`, `calls=17466`, `avg_iters_per_call=1.38984`

## Decomposed Timing Breakdown

- `Newton polish`: `2.647 s`
- `Lambert-W`: `2.858 s`
- `Lambert other`: `1.458 s`
- `Outside updater`: `1.159 s`
- `b coefficient`: `0.914 s`
- `a coefficient`: `0.605 s`
- `Other updater`: `0.152 s`
- `Lambert total` is intentionally omitted from the figure and table because it would double count the Lambert-W, Newton-polish, and Lambert-other subcomponents.

## Comparison To Float32 Overrelaxed

- Wall time: `21.953 s` -> `9.794 s` (`2.242x` speedup)
- Outer iterations are matched: `16.175` vs `16.172`.
- Newton iterations per call drop from `6.867` to `1.390`.
- Newton polish time drops from `12.446 s` to `2.647 s`.

## Files

- `timing_breakdown.json`: machine-readable summary with raw totals and presentation components
- `component_timing.csv`: cleaned per-component table without Lambert total
- `component_timing_comparison.csv`: float32 vs float64 component table
- `component_timing_bar.png` / `.pdf`: paper-ready float64 component figure
- `component_timing_comparison_bar.png` / `.pdf`: paper-ready float32 vs float64 comparison figure
