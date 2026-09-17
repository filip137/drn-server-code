# Figure-6/OM RESET-only hidden-SET screen

This exploratory matched screen reuses the frozen `784-256-10` teacher,
endpoint seeds, OM identities, write seeds, data split, solver, HWA learning
rates, and logit gain from the completed exact-endpoint ladder. Its only
intended mapping intervention is the information boundary.

The target mapper knows each cell's exact fixed Figure-6 RESET endpoint but
does not receive its true SET endpoint. Within every logical four-cell quad it
requests one shared baseline equal to the largest RESET and one common active
conductance based on the predeclared nominal SET value `2.0`. The hidden true
SET can saturate a request but may not resize it. Program-and-verify observes
absolute apparent conductance rather than true-SET-normalized progress.

Run the focused direct-versus-HWA screen with:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_relu_drn.figure6_om_reset_only_screen \
  --config examples/mnist_relu_drn/figure6_om_784_256_10_reset_only/shared_quad_nominal_set.json \
  --output-root simulation_results/exploratory_noncanonical_figure6_om_784_256_10_reset_only
```

Every device-facing forward uses the current held apparent state. Persistent
state is write authority and a secondary diagnostic only. This first screen
does not run pulse-mediated Adam: its gradient/update information boundary
must be specified separately if the RESET-only mapping passes deployment.

The completed exploratory result and its interpretation are recorded in
[`docs/figure6_om_784_256_10_reset_only_results.md`](../../../docs/figure6_om_784_256_10_reset_only_results.md).
