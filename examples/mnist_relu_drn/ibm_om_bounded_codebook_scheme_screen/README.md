# IBM OM deterministic bounded-codebook scheme screen

This strict screen contract is consumed by
`experiments/mnist_relu_drn/ibm_om_bounded_codebook_scheme_screen.py`.
The frozen ReLU teacher and the ordinary MNIST DRN model config remain
explicit command-line inputs. The immutable initial-RESET calibration receipt
is also an explicit command-line input; this file pins its SHA-256 and values.

The source checkpoint and solver use FP32. Each requested conductance is
bounded by its assigned AIHWKit OM identity and projected to the nearest state
on a deterministic lower-to-SET trajectory with both stochastic pulse terms
disabled. This is an oracle codebook endpoint control, not program-and-verify.

The four arms cross fixed-reference use with four versus eight physical
devices per original signed weight. Every arm and both endpoint controls reuse
the initial-RESET fractions `[1.0, 1.0]` and positive output gain
`4.46683592150963`; no scheme is separately calibrated. Development assignment
`86001` supplies diagnostics only. Assignments `87001`, `87002`, and `87003`
are evaluated with the shared calibration frozen.
