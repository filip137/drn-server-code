# Initialization gradient quality

[Conv3: cosine to BPTT only](conv3_cosine_to_bptt_vs_noise.jpg)

[Cosine to BPTT across layers](cosine_to_bptt_across_layers.jpg) · [Cosine to clean EqProp across layers](cosine_to_clean_ep_across_layers.jpg)

The four layer-profile columns use eta=1e-6, 1e-5, 1e-4 and 3e-4.

Noise sweeps, with cosine and relative noise-error norm: [Conv1](conv1_gradient_quality_vs_noise.jpg) · [Conv2](conv2_gradient_quality_vs_noise.jpg) · [Conv3](conv3_gradient_quality_vs_noise.jpg).

Saved exp013 stage-A summaries, at matched initial output D/F=1/4/6 for Conv1/2/3. The 14 nonzero noise levels use 192 examples (12 batches of 16), three acquisition draws and one model initializer. Every noninput layer uses the same relative coefficient eta. No training or new replay. Stages are not pooled, and uncertainty over model seeds is unavailable. Curves are means of per-batch metrics; bands span the three draw means, not confidence intervals. Finite-K BPTT is the reference, not a claimed exact equilibrium gradient.

Relative noise error is ||g_noisy_EP - g_clean_EP|| / ||g_clean_EP||. This isolates acquisition perturbation; cosine to BPTT also includes any clean EqProp/BPTT discrepancy. The separate clean-EqProp cosine plot makes that distinction visible. Noise-sweep norm-error panels have individually scaled logarithmic y-axes; their horizontal line at 1 means error norm equals clean-gradient norm.

The accompanying CSV contains all 405 selected summary rows, including 27 eta=0 controls evaluated on the original 576-example clean cohort. Zero-noise rows are retained for provenance but are not plotted on the logarithmic noise axis or pooled with the 192-example noisy estimates. Original source: results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_a/layer_summary.csv.

Reproduce: `python -m experiments.plot_layer_gradient_quality --source-csv campaigns/read-noise-beta/series/006-relative-endpoint-noise/results/figures/exp013-gradient-quality/plotted_values.csv --output-dir campaigns/read-noise-beta/series/006-relative-endpoint-noise/results/figures/exp013-gradient-quality`.
