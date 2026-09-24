# Conv3 cosine transition across T

Filip requested the T values at which the noisy EP–BPTT cosine changes for
ours at training/read sigma 5e-4, then specified exactly `T=[12,16,20,32]`.
Replay only that final epoch-30, seed-0 A100 checkpoint at its unchanged
training injected beta `5.26875648112`, with K=8. Reuse the existing T8 and
T64 training-beta cells as anchors. This is an exploratory cosine diagnostic;
residuals do not select, exclude or gate points. No additional adaptive T
grid, beta sweep, training, accuracy evaluation or official-test read is planned.

Keep the same 36 batches of 16 ordinary-MNIST validation examples, all four
saved read-noise seeds, clean controls, float64 centered frozen-current EP,
explicit perfect-diode dictionaries, exact-zero biases, input gain 360,
paired 20-output squared loss, and unchanged checkpoint weights. BPTT
differentiates exactly K zero-nudge steps from each T's post-free state.
Report layerwise median cosine and the 10–90% distribution, and distinguish
the first positive sampled readout cosine from a claim about unmeasured T.

Four readable configs are under
`configs/conv/eqprop_conv3_cosine_t_transition_20260922_v1/T{12,16,20,32}.json`.
Run each directly with `python -m experiments.replay_conv3_trained_beta_noise
--config CONFIG --data-root /home/filiposana/datasets/mnist`. Run each same
command first with `--smoke` for its one-batch semantic check, then production.
The existing runner only needed its fixed six-checkpoint assertion generalized
to the declared positive checkpoint count; its scientific computations remain
unchanged. All four configs each declare one checkpoint and one beta factor.

Target: one Akib RTX3080 worker, SSH alias `akibscomputer`, Python
`/home/filiposana/miniconda3/envs/py312/bin/python`, detached through the Akib
launcher. The current local, nom-cool-1, Trex, Riri, Fifi and Loulou GPUs are
occupied; Jean Zay has no user job, but Akib is the leaner available target
and matches both cached anchors' numerical environment. Check occupancy again
immediately before launch and preserve unrelated work. Expected duration is
about six minutes; cap 225 seconds per T including its smoke and 900 seconds
overall. Monitor semantic batch/cell progress each minute and retain exit/log
tracking. No additional GPU is allocated.

Remote outer directory:
`/home/filiposana/server_code/results/eqprop-conv3-cosine-t-transition-20260922-v1`.
Use an isolated workspace and launcher directory under that root. Authoritative
local output:
`results/eqprop-conv3-cosine-t-transition-20260922-v1/`.
Expected new coverage: four production bundles, four smoke bundles, 2,880
production layer comparisons. Collect and validate all outputs before final
interpretation, then generate a matplotlib T plot and CSV including the two
cached anchors and record the conclusion in the experimental manifest.

## Completed result

Akib nohup supervisor 483697 completed all four smokes and all four requested
production T values with exit 0. The worker released the GPU. Runtime was
363.18 seconds for replay including smokes, 394 seconds of launcher wall time,
within the 900-second cap. All eight bundles validate locally, all 136 output
file hashes match Akib, and all four source checkpoint files remain unchanged.
There are no failed, missing or excluded scientific cases.

Noisy readout median cosines at T8/12/16/20/32/64 are
`[-.29315,.61105,.68455,.73593,.82046,.88544]`. T12 is the first positive
sampled median; T20 is the first sampled value with all 144 readout comparisons
positive. T9–11 were not measured. Readout cosine still changes by .06498
between T32 and T64. Conv3 is non-monotonic, with its largest sampled median
at T12. Clean readout controls show the same T dependence.

[Curated interpretation](experimental_manifest.md#conv3-readout-cosine-transition-by-t12-september-22)
· [Report and plot](../results/eqprop-conv3-cosine-t-transition-20260922-v1/analysis/report.md)
· [Validation](../results/eqprop-conv3-cosine-t-transition-20260922-v1/collection_validation.json).
