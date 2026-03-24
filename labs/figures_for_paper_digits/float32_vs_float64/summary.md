# Float32 vs Float64 Timing Comparison

Canonical `double_diode_exponential` CPU timing runs, comparing old float32 `overrelaxed` vs new float64 `float64_timed_overrelaxed`.

- Points compared: 14
- Total float32 time: 203.617 s
- Total float64 time: 131.178 s
- Overall speedup: 1.552x
- Mean per-point speedup: 1.549x
- Median per-point speedup: 1.606x
- Float32 faster cases: 1H/128, 2H/64
- Float64 faster cases: 1H/64, 1H/256, 1H/512, 1H/1024, 2H/128, 2H/256, 2H/512, 2H/1024, 3H/64, 3H/128, 3H/256, 3H/512

Best float64 win:
- 3H/256: 19.585 s -> 10.015 s (1.956x)

Float32-favored case:
- 1H/128: 1.917 s -> 2.004 s (0.957x)
