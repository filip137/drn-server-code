# Table 3: CPU-only Runs Without SPICE Baselines

| nonlinearity | hidden layers | hidden size | device | batch size | CD runtime [s] | validation accuracy [%] | average outer iterations | note |
|---|---:|---:|---|---:|---:|---:|---:|---|
| double_diode_exponential | 2 | 1024 | CPU | 1 | 20.972 | 86.11 | 31.733 | Missing SPICE states/timing. |
| double_diode_exponential | 3 | 512 | CPU | 1 | 34.216 | 91.94 | 51.925 | Missing SPICE states/timing. |
| experimental | 2 | 1024 | CPU | 1 | 22.307 | 86.11 | 68.503 | Missing SPICE states/timing. |
| experimental | 3 | 512 | CPU | 1 | 19.390 | 91.94 | 69.717 | Missing SPICE states/timing. |
| single_diode_exponential | 2 | 512 | CPU | 1 | 17.040 | 83.61 | 31.778 | Missing SPICE states/timing. |
| single_diode_exponential | 2 | 1024 | CPU | 1 | 19.923 | 93.06 | 27.586 | Missing SPICE states/timing. |
| single_diode_exponential | 3 | 256 | CPU | 1 | 77.134 | 84.44 | 67.850 | No SPICE timing available; sourced from simulation_results validation/from_20260318-115323_model; previously skipped from matched SPICE tables. |
| single_diode_exponential | 3 | 512 | CPU | 1 | 13.392 | 83.06 | 11.581 | Missing SPICE states/timing. |
