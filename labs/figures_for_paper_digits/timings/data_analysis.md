# Validation State Saturation Analysis

## Scope

This note analyzes saturation in `validation_states.npz` for the `double_diode_exponential` depth-3 (`hidden_3`) runs under `/home/filip/server_code/labs/figures_for_paper_digits/timings`.

The saturation criterion is:

- a neuron is counted as saturated when `abs(v) > 0.8`

The four runs analyzed are:

- [hidden_64 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_64/20260311-185700_double_diode_exponential/validation_metadata.json)
- [hidden_128 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_128/20260313-155212_double_diode_exponential/validation_metadata.json)
- [hidden_256 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_256/validation/20260317-161243_double_diode_exponential/validation_metadata.json)
- [hidden_512 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_512/validation/20260311-185941_double_diode_exponential/validation_metadata.json)

For each run, I compared:

- `newton_iteration_stats.avg_iters_per_call`
- `newton_iteration_stats.calls`
- the fraction of saturated neurons in each layer
- the mean number of saturated neurons per validation sample

## Newton Summary

| width | avg inner Newton iters/call | Newton calls | avg outer iters |
|---|---:|---:|---:|
| 64 | 7.670 | 15,222 | 9.819 |
| 128 | 6.867 | 17,328 | 16.044 |
| 256 | 7.765 | 17,727 | 16.414 |
| 512 | 3.333 | 55,908 | 51.767 |

## Saturation By Layer

### Width 64

Source: [validation_states.npz](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_64/20260311-185700_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.729861 | 46.711 / 64 | 47 | 50 | 54 |
| Layer_2 | 0.892578 | 57.125 / 64 | 57 | 62 | 64 |
| Layer_3 | 1.000000 | 64.000 / 64 | 64 | 64 | 64 |
| Layer_4 | 1.000000 | 20.000 / 20 | 20 | 20 | 20 |

### Width 128

Source: [validation_states.npz](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_128/20260313-155212_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.139388 | 17.842 / 128 | 18 | 22 | 31 |
| Layer_2 | 0.038802 | 4.967 / 128 | 4 | 9 | 22 |
| Layer_3 | 0.943967 | 120.828 / 128 | 120 | 128 | 128 |
| Layer_4 | 1.000000 | 20.000 / 20 | 20 | 20 | 20 |

### Width 256

Source: [validation_states.npz](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_256/validation/20260317-161243_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.045074 | 11.539 / 256 | 11 | 19 | 27 |
| Layer_2 | 0.004058 | 1.039 / 256 | 1 | 3 | 9 |
| Layer_3 | 1.000000 | 256.000 / 256 | 256 | 256 | 256 |
| Layer_4 | 1.000000 | 20.000 / 20 | 20 | 20 | 20 |

### Width 512

Source: [validation_states.npz](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_3/hidden_512/validation/20260311-185941_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.000000 | 0.000 / 512 | 0 | 0 | 0 |
| Layer_2 | 0.000000 | 0.000 / 512 | 0 | 0 | 0 |
| Layer_3 | 0.000000 | 0.000 / 512 | 0 | 0 | 0 |
| Layer_4 | 0.000000 | 0.000 / 20 | 0 | 0 | 0 |

## Conclusions

The saturation view is much clearer than the raw min/max view.

- Width 64, 128, and 256 all have very strong saturation in deeper layers.
- Width 512 has no saturation at all under the `abs(v) > 0.8` threshold.

The strongest pattern is:

- Width 64:
  - `Layer_1` and `Layer_2` are already heavily saturated.
  - `Layer_3` and `Layer_4` are fully saturated.
- Width 128:
  - `Layer_1` and `Layer_2` are only lightly saturated.
  - `Layer_3` is almost fully saturated.
  - `Layer_4` is fully saturated.
- Width 256:
  - `Layer_1` and `Layer_2` are barely saturated.
  - `Layer_3` is fully saturated.
  - `Layer_4` is fully saturated.
- Width 512:
  - no layer has any saturated neurons at all.

This lines up with the Newton behavior in an informative way:

- Widths 64, 128, and 256 all have substantial saturation somewhere in the deep hidden state, and all three have similar average inner Newton cost: about `6.9` to `7.8` iterations per call.
- Width 512 has zero saturated neurons in every internal layer and also has the lowest inner Newton cost: `3.333` iterations per call.

So the main conclusion is:

- saturation appears to be a better explanatory signal than raw global min/max voltage ranges for inner Newton cost
- the important distinction is not a smooth monotonic trend with width
- the key split is between the saturated runs (`64`, `128`, `256`) and the unsaturated run (`512`)

There is still an important caveat:

- Width 512 is cheaper per Newton call, but it is not cheaper overall.
- It has far more outer equilibrium iterations and far more total Newton calls.

Practical takeaway:

- For these depth-3 double-diode runs, saturation seems related to how expensive each Newton solve is.
- But the total workload is still governed by both inner Newton cost and outer equilibrium dynamics.

## Hidden 2 Saturation Analysis

This section repeats the same `abs(v) > 0.8` saturation analysis for the `double_diode_exponential` depth-2 (`hidden_2`) runs.

The runs analyzed are:

- [hidden_64 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_2/hidden_64/20260311-170549_double_diode_exponential/validation_metadata.json)
- [hidden_128 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_2/hidden_128/super_high_accuracy/20260316-145914_double_diode_exponential/validation_metadata.json)
- [hidden_256 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_2/hidden_256/20260311-170612_double_diode_exponential/validation_metadata.json)
- [hidden_512 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_2/hidden_512/20260311-170652_double_diode_exponential/validation_metadata.json)
- [hidden_1024 metadata](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_2/hidden_1024/20260311-170758_double_diode_exponential/validation_metadata.json)

For `hidden_128`, the metadata’s original `validation_states` path no longer exists under `simulation_results`, so the analysis used the archived copy next to the metadata under `timings`.

### Newton Summary

| width | avg inner Newton iters/call | Newton calls | avg outer iters |
|---|---:|---:|---:|
| 64 | 0.000 | 0 | 16.742 |
| 128 | 6.888 | 11,408 | 15.844 |
| 256 | 0.000 | 0 | 18.031 |
| 512 | 0.000 | 0 | 26.464 |
| 1024 | 0.000 | 0 | 31.656 |

### Saturation By Layer

#### Width 64

Source: [validation_states.npz](/home/filip/server_code/simulation_results/digits_medium_network/hidden_2/double_diode_exponential/hidden_64/validation/20260311-170549_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.000000 | 0.000 / 64 | 0 | 0 | 0 |
| Layer_2 | 0.000000 | 0.000 / 64 | 0 | 0 | 0 |
| Layer_3 | 0.000000 | 0.000 / 20 | 0 | 0 | 0 |

#### Width 128

Source: [validation_states.npz](/home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/hidden_2/hidden_128/super_high_accuracy/20260316-145914_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.036328 | 4.650 / 128 | 4 | 9 | 15 |
| Layer_2 | 0.993403 | 127.156 / 128 | 128 | 128 | 128 |
| Layer_3 | 1.000000 | 20.000 / 20 | 20 | 20 | 20 |

#### Width 256

Source: [validation_states.npz](/home/filip/server_code/simulation_results/digits_medium_network/hidden_2/double_diode_exponential/hidden_256/validation/20260311-170612_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.000000 | 0.000 / 256 | 0 | 0 | 0 |
| Layer_2 | 0.000000 | 0.000 / 256 | 0 | 0 | 0 |
| Layer_3 | 0.000000 | 0.000 / 20 | 0 | 0 | 0 |

#### Width 512

Source: [validation_states.npz](/home/filip/server_code/simulation_results/digits_medium_network/hidden_2/double_diode_exponential/hidden_512/validation/20260311-170652_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.000000 | 0.000 / 512 | 0 | 0 | 0 |
| Layer_2 | 0.000000 | 0.000 / 512 | 0 | 0 | 0 |
| Layer_3 | 0.000000 | 0.000 / 20 | 0 | 0 | 0 |

#### Width 1024

Source: [validation_states.npz](/home/filip/server_code/simulation_results/digits_medium_network/hidden_2/double_diode_exponential/hidden_1024/validation/20260311-170758_double_diode_exponential/validation_states.npz)

| layer | saturated fraction | mean saturated / sample | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|
| Layer_1 | 0.000000 | 0.000 / 1024 | 0 | 0 | 0 |
| Layer_2 | 0.000000 | 0.000 / 1024 | 0 | 0 | 0 |
| Layer_3 | 0.000000 | 0.000 / 20 | 0 | 0 | 0 |

### Conclusions For Hidden 2

The `hidden_2` case is much less ambiguous than `hidden_3`.

- Width `128` is the only run with any Newton activity at all.
- Width `128` is also the only run with any saturation under the `abs(v) > 0.8` threshold.
- Widths `64`, `256`, `512`, and `1024` have:
  - zero saturated neurons in every layer
  - zero Newton calls
  - zero average inner Newton iterations per call

For width `128`, the saturation is very concentrated:

- `Layer_1` is only lightly saturated
- `Layer_2` is almost fully saturated
- `Layer_3` is fully saturated

So for `hidden_2`, the relationship is stronger than for `hidden_3`:

- saturation and Newton usage essentially appear together in only one run
- no-saturation runs coincide with no Newton inner iterations

That does not prove causality by itself, but in this `hidden_2` slice the saturation metric is an excellent discriminator for whether Newton is active at all.
