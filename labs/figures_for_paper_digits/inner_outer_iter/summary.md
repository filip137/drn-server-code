# Inner / Outer Iteration Analysis

Generated at `2026-03-20T14:39:32`.

Definitions:
- outer iterations = `equilibrium_iterations.avg_iterations`
- inner iterations = Newton iterations from `newton_iteration_stats`
- `inner_iterations_per_outer_iteration = total_inner_iterations / total_outer_iterations`

## single_diode_exponential

- highest outer-iteration load: depth 3, width 256: outer=228.644, inner/sample=1396.372, inner/outer=6.107
- highest inner-per-outer load: depth 3, width 128: outer=17.031, inner/sample=178.814, inner/outer=10.500
- highest total inner work per sample: depth 3, width 256: outer=228.644, inner/sample=1396.372, inner/outer=6.107
- lowest nonzero inner-per-outer load: depth 1, width 64: outer=5.544, inner/sample=22.178, inner/outer=4.000
- zero-inner-Newton cases: 1H/512

- CSV: `single_diode_exponential_inner_outer_metrics.csv`
- Plot: `single_diode_exponential_inner_outer_iterations.png`

| depth | width | outer_avg_iterations | inner_avg_iterations_per_call | inner_iterations_per_sample | inner_iterations_per_outer_iteration |
|---|---:|---:|---:|---:|---:|
| 1 | 64 | 5.544444 | 2.000000 | 22.177778 | 4.000000 |
| 1 | 128 | 9.000000 | 2.000000 | 36.000000 | 4.000000 |
| 1 | 256 | 5.872222 | 2.000000 | 23.488889 | 4.000000 |
| 1 | 512 | 5.475000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 1024 | 5.108333 | 2.000000 | 20.433333 | 4.000000 |
| 2 | 64 | 15.169444 | 1.400430 | 84.975000 | 5.601721 |
| 2 | 128 | 14.877778 | 1.083271 | 64.466667 | 4.333084 |
| 2 | 256 | 21.461111 | 1.000000 | 85.844444 | 4.000000 |
| 2 | 512 | 31.777778 | 1.000000 | 127.111111 | 4.000000 |
| 2 | 1024 | 27.586111 | 1.000000 | 110.344444 | 4.000000 |
| 3 | 64 | 57.675000 | 1.143115 | 395.575000 | 6.858691 |
| 3 | 128 | 17.030556 | 1.749932 | 178.813889 | 10.499592 |
| 3 | 256 | 228.644444 | 1.017863 | 1396.372222 | 6.107178 |
| 3 | 512 | 11.580556 | 1.606820 | 111.647222 | 9.640921 |

## double_diode_exponential

- highest outer-iteration load: depth 3, width 512: outer=51.936, inner/sample=519.361, inner/outer=10.000
- highest inner-per-outer load: depth 3, width 64: outer=9.556, inner/sample=331.544, inner/outer=34.697
- highest total inner work per sample: depth 3, width 512: outer=51.936, inner/sample=519.361, inner/outer=10.000
- lowest nonzero inner-per-outer load: depth 1, width 128: outer=8.969, inner/sample=8.981, inner/outer=1.001

- CSV: `double_diode_exponential_inner_outer_metrics.csv`
- Plot: `double_diode_exponential_inner_outer_iterations.png`

| depth | width | outer_avg_iterations | inner_avg_iterations_per_call | inner_iterations_per_sample | inner_iterations_per_outer_iteration |
|---|---:|---:|---:|---:|---:|
| 1 | 64 | 9.000000 | 7.000000 | 63.000000 | 7.000000 |
| 1 | 128 | 8.969444 | 1.001239 | 8.980556 | 1.001239 |
| 1 | 256 | 9.000000 | 7.000000 | 63.000000 | 7.000000 |
| 1 | 512 | 9.000000 | 7.000000 | 63.000000 | 7.000000 |
| 1 | 1024 | 9.000000 | 7.000000 | 63.000000 | 7.000000 |
| 2 | 64 | 16.747222 | 1.002156 | 33.566667 | 2.004312 |
| 2 | 128 | 15.841667 | 6.880501 | 217.997222 | 13.761003 |
| 2 | 256 | 18.030556 | 3.318518 | 119.669444 | 6.637036 |
| 2 | 512 | 26.536111 | 4.000000 | 212.288889 | 8.000000 |
| 2 | 1024 | 31.733333 | 4.000000 | 253.866667 | 8.000000 |
| 3 | 64 | 9.555556 | 7.677602 | 331.544444 | 34.696512 |
| 3 | 128 | 16.175000 | 6.867365 | 333.238889 | 20.602095 |
| 3 | 256 | 16.425000 | 7.764361 | 382.588889 | 23.293083 |
| 3 | 512 | 51.936111 | 3.333333 | 519.361111 | 10.000000 |

