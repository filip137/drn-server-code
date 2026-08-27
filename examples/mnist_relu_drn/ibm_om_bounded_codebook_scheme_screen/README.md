# IBM OM deterministic bounded-codebook scheme screen

This strict screen contract is consumed by
`experiments/mnist_relu_drn/ibm_om_bounded_codebook_scheme_screen.py`.
The frozen ReLU teacher and the ordinary MNIST DRN model config remain
explicit command-line inputs; this file fixes only the device-aware screen.

The source checkpoint and solver use FP32. Each requested conductance is
bounded by its assigned AIHWKit OM identity and projected to the nearest state
on a deterministic lower-to-SET trajectory with both stochastic pulse terms
disabled. This is an oracle codebook endpoint control, not program-and-verify.

The four arms cross fixed-reference use with four versus eight physical
devices per original signed weight. Development assignment `86001` selects
the scale fractions. Assignments `87001`, `87002`, and `87003` are evaluated
with those choices frozen.
