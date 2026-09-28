# Initialization displacement across layers

[All architectures](conv123_absolute_and_fractional_displacement.jpg) · [Conv1](conv1_absolute_and_fractional_displacement.jpg) · [Conv2](conv2_absolute_and_fractional_displacement.jpg) · [Conv3](conv3_absolute_and_fractional_displacement.jpg)

Conv1/2/3 use output D/F targets 1/4/6, respectively, matching the selected training settings at initialization. Original source: results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_a/physical_state_controls.csv. These physical-state measurements use 576 examples; the stage-A noisy-gradient screen's smaller cohort is not used here. No stages are pooled.

D is RMS((v_plus-v_minus)/2); P is the pooled RMS of both clean nudged endpoints. Each row represents one layer and scheme, pooled over the cohort and nodes. Both y-axes are logarithmic with independently fitted limits. Lines connect discrete layers; one initializer, no seed uncertainty. Baseline uses hollow green markers so its near-overlap with legacy remains visible. No noise is injected or simulation performed by this plotting script.

[Plotted measurements](plotted_values.csv). Reproduce from the repository root: `python -m experiments.plot_layer_fractional_displacement --source-csv campaigns/read-noise-beta/series/006-relative-endpoint-noise/results/figures/exp013-layer-displacement/plotted_values.csv --output-dir campaigns/read-noise-beta/series/006-relative-endpoint-noise/results/figures/exp013-layer-displacement`.
