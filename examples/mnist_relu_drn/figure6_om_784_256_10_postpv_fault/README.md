# Exploratory 784-256-10 Figure-6 / IBM-OM ladder

`full.json` is a fail-closed protocol for the eight matched arms requested in
the large experiment. The evaluated model is a perfect-diode DRN with physical
dimensions `[1568, 512, 20]` (813,056 four-cell dual-rail devices), distilled
from a bias-free ReLU `[784, 256, 10]` teacher.

Every hardware-facing forward uses the current held apparent state. Persistent
state is used only as the pulse-update authority and as a clearly labelled
secondary diagnostic. Off-chip HWA uses static sampled endpoint views and
updates a digital logical master. P&V always runs on the repaired plant first;
the corrupted branch is an exact clone followed by a synthetic RESET-stuck
intervention on the paired published-OM mask.

The resumable process graph is launched with:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_relu_drn.figure6_om_256_ladder \
  --config examples/mnist_relu_drn/figure6_om_784_256_10_postpv_fault/full.json \
  --campaign-root simulation_results/exploratory_noncanonical_figure6_om_784_256_10_postpv_fault \
  launch
```

The launcher trains the teacher if an accepted campaign checkpoint is absent,
then executes `prepare`, nine `hwa-screen` candidates plus selection,
`hwa-train`, nine `deploy` replicas, the two-family `adam-screen`, 36
`adam-arm` jobs, and `summarize`. Completed jobs are skipped on a relaunch;
failed attempts remain preserved. A full launch refuses a dirty Git worktree
so every run has a frozen source revision. Reduced `--smoke` output remains
`exploratory_noncanonical` and is not final evidence.

The final summary first averages the three write streams within each physical
array, then reports the mean and full range across the three array means. The
HWA and scratch Adam families select separate learning rates, so their final
comparison is explicitly best-tuned per initialization rather than a
same-learning-rate causal contrast.

The completed exploratory accuracy/KL analysis and the exact qualification of
the open-loop pulse-mediated Adam writer are in
[`figure6_om_784_256_10_postpv_fault_results.md`](../../../docs/figure6_om_784_256_10_postpv_fault_results.md).
