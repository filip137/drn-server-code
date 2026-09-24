# Conductance occupancy figure for Section 4.2

Generated 2026-09-18 from the same 81 bounded BPTT runs used in the current
manuscript accuracy table. The figure shows the fraction of stored trainable
conductance coefficients exactly at each bound, evaluated at each run's
best-validation checkpoint.

[PDF](figures/bounded_conductance_occupancy_best.pdf) ·
[PNG](figures/bounded_conductance_occupancy_best.png) ·
[JPG](figures/bounded_conductance_occupancy_best.jpg) ·
[SVG](figures/bounded_conductance_occupancy_best.svg)

The columns are Conv1, Conv2 and Conv3; rows separate G_min and G_max.
Colors and marker shapes identify the amplification schemes. Points are
means across seeds 0, 1 and 2; error bars are sample standard deviations.
The legend uses "Balanced" for the (4,1) scheme, matching the latest
Overleaf manuscript; its source-data identifier remains `ours`.
The x-axis is logarithmic. Each row shares a vertical scale across depths;
the upper- and lower-bound rows use different scales.

At every tested depth and ceiling, mean lower-bound occupancy is ordered
baseline > ours > legacy. At G_max = 1e-4, the measurements are:

| Architecture | Scheme | At G_min (%) | At G_max (%) |
|---|---|---:|---:|
| Conv1 | Baseline | 34.25 ± 3.03 | 2.33 ± 0.05 |
| Conv1 | Ours | 22.68 ± 1.15 | 0.95 ± 0.07 |
| Conv1 | Legacy | 9.78 ± 0.62 | 0.92 ± 0.09 |
| Conv2 | Baseline | 39.22 ± 1.39 | 5.58 ± 0.69 |
| Conv2 | Ours | 26.69 ± 3.75 | 2.78 ± 0.51 |
| Conv2 | Legacy | 22.30 ± 1.69 | 1.32 ± 0.05 |
| Conv3 | Baseline | 63.99 ± 1.93 | 7.63 ± 0.45 |
| Conv3 | Ours | 38.32 ± 3.04 | 3.33 ± 0.59 |
| Conv3 | Legacy | 28.58 ± 6.14 | 2.55 ± 0.80 |

Upper-bound occupancy decreases with increasing ceiling for every
architecture/scheme. Lower-bound occupancy generally decreases too, with
a small exception for Conv1 ours between the two largest ceilings. The
lower-bound ordering accompanies the reverse accuracy ordering in the
existing table, but does not establish why accuracy differs: learning-rate
vectors and amplification both differ across schemes. A single checkpoint
measures occupancy, not persistent sticking or optimizer clipping frequency.

Suggested caption:

> Conductance occupancy at the bounds after bounded BPTT training. Columns
> show Conv1, Conv2 and Conv3; the top and bottom rows report the percentages
> of weights exactly at G_min and G_max, respectively, as G_max varies with
> G_min fixed at 1e-5. Colors identify the baseline (A,B)=(1,1), balanced (4,1)
> and legacy (4,0.25) amplification schemes. Points and error bars give the
> mean and sample standard deviation across three seeds, using the same
> best-validation checkpoints as the bounded-conductance accuracy table.
> Percentages pool all convolutional and dense conductance coefficients,
> excluding biases; each stored coefficient is counted once. Vertical scales
> are shared across architectures within each row and differ between rows.

The original shared T/K = 4/4, 6/6 and 8/8 is retained. Endpoint equality
uses bounds cast to each checkpoint array's dtype; no tolerance threshold
is used. The main figure weights tensors by their numbers of coefficients,
so it is a network-wide percentage rather than an equally weighted layer
average. The layer CSV retains all 243 tensor measurements.

Verification covered all 81 inclusion rows, result/manifest/metrics hashes,
indexed config and best-weight hashes, successful completion, model seeds,
amplification, bounds, T/K, tensor identity, finite in-range weights and
agreement with the source accuracy ledger. This selected-input audit did not
rehash unrelated checkpoints or other bundle artifacts. No training,
checkpoint modification, dataset loading or official-test access occurred.

- [27 group means and sample SDs](bounded_conductance_occupancy_best_summary.csv)
- [81 per-seed measurements and source hashes](bounded_conductance_occupancy_best_per_seed.csv)
- [243 per-layer measurements](bounded_conductance_occupancy_best_per_layer.csv)
- [Measurement provenance](bounded_conductance_occupancy_best_provenance.json)
- [Reproduction script](plot_bounded_conductance_occupancy.py)

Reproduce in an environment with NumPy and Matplotlib:

```bash
python plot_bounded_conductance_occupancy.py
```
