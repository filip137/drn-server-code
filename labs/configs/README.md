# Configuration status

Files with the `small_network_*.json` naming convention use the retired flat
research-runner schema. They remain tracked as provenance for historical and
paper results, but `python -m ebl` does not interpret them.

New and rerun experiments must use a strict, nested, versioned configuration
such as [`../../examples/small_drn/base.json`](../../examples/small_drn/base.json).
Convert a historical case deliberately by mapping each scientific setting;
do not add a permissive compatibility parser or silently fill missing diode
parameters.

The other files in this directory belong to still-live MNIST/Fashion-MNIST
tools and retain their own schemas.
