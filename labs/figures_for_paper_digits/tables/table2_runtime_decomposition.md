# Table 2: Runtime Decomposition

| nonlinearity | hidden layers | hidden size | CD runtime [s] | SPICE runtime [s] | speedup vs SPICE | average outer iterations | average inner iterations per call | total inner iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| double_diode_exponential | 1 | 64 | 2.575 | 22.065 | 8.569 | 9.000 | 1.889 | 6119 |
| double_diode_exponential | 1 | 128 | 2.004 | 73.608 | 36.734 | 8.969 | 1.000 | 3229 |
| double_diode_exponential | 1 | 256 | 2.831 | 100.499 | 35.495 | 9.000 | 1.966 | 6369 |
| double_diode_exponential | 1 | 512 | 3.051 | 309.581 | 101.468 | 9.000 | 1.996 | 6468 |
| double_diode_exponential | 1 | 1024 | 3.580 | 1107.349 | 309.344 | 9.000 | 2.000 | 6480 |
| double_diode_exponential | 2 | 64 | 5.844 | 68.514 | 11.724 | 16.753 | 1.000 | 12062 |
| double_diode_exponential | 2 | 128 | 6.143 | 549.956 | 89.522 | 15.836 | 1.212 | 13816 |
| double_diode_exponential | 2 | 256 | 7.275 | 4312.234 | 592.736 | 18.031 | 1.000 | 12982 |
| double_diode_exponential | 2 | 512 | 12.041 | 51768.450 | 4299.287 | 26.544 | 1.000 | 19112 |
| double_diode_exponential | 3 | 64 | 10.936 | 168.090 | 15.370 | 9.556 | 1.715 | 26737 |
| double_diode_exponential | 3 | 128 | 9.694 | 1183.510 | 122.090 | 16.172 | 1.390 | 24275 |
| double_diode_exponential | 3 | 256 | 10.015 | 14361.823 | 1434.046 | 16.422 | 1.288 | 22849 |
| experimental | 1 | 64 | 5.306 | 22.023 | 4.151 | 13.781 | 20.353 | 100969 |
| experimental | 1 | 128 | 1.386 | 39.071 | 28.193 | 14.147 | 1.024 | 5213 |
| experimental | 1 | 256 | 3.799 | 94.790 | 24.952 | 8.047 | 20.874 | 60471 |
| experimental | 1 | 512 | 3.992 | 311.140 | 77.938 | 7.233 | 21.082 | 54898 |
| experimental | 1 | 1024 | 4.736 | 1094.952 | 231.178 | 5.758 | 24.475 | 50737 |
| experimental | 2 | 64 | 4.446 | 61.481 | 13.827 | 33.400 | 1.000 | 24048 |
| experimental | 2 | 128 | 5.389 | 543.731 | 100.902 | 21.242 | 3.956 | 60509 |
| experimental | 2 | 256 | 6.144 | 4301.242 | 700.089 | 39.950 | 1.000 | 28764 |
| experimental | 2 | 512 | 11.829 | 52009.107 | 4396.647 | 61.986 | 1.000 | 44630 |
| experimental | 3 | 64 | 6.900 | 165.079 | 23.924 | 7.425 | 15.819 | 126854 |
| experimental | 3 | 128 | 12.900 | 1131.481 | 87.714 | 26.236 | 6.905 | 195665 |
| experimental | 3 | 256 | 5.348 | 13723.796 | 2566.096 | 13.167 | 3.643 | 51802 |
| single_diode_exponential | 1 | 64 | 2.157 | 22.600 | 10.477 | 5.544 | 2.000 | 7984 |
| single_diode_exponential | 1 | 128 | 3.640 | 30.891 | 8.487 | 9.000 | 2.000 | 12960 |
| single_diode_exponential | 1 | 256 | 2.608 | 69.386 | 26.602 | 5.872 | 2.000 | 8456 |
| single_diode_exponential | 1 | 512 | 2.525 | 198.342 | 78.563 | 5.472 | 1.993 | 7853 |
| single_diode_exponential | 1 | 1024 | 2.621 | 793.847 | 302.880 | 5.108 | 2.000 | 7356 |
| single_diode_exponential | 2 | 64 | 7.806 | 76.661 | 9.821 | 15.169 | 1.400 | 30591 |
| single_diode_exponential | 2 | 128 | 7.582 | 477.836 | 63.025 | 14.878 | 1.083 | 23208 |
| single_diode_exponential | 2 | 256 | 10.335 | 4293.154 | 415.397 | 21.461 | 1.000 | 30904 |
| single_diode_exponential | 3 | 64 | 37.486 | 172.434 | 4.600 | 57.675 | 1.143 | 142407 |
| single_diode_exponential | 3 | 128 | 15.560 | 1660.868 | 106.738 | 17.031 | 1.750 | 64373 |
