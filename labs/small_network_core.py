"""Import-error shim for the archived ``small_network_core`` module."""

raise ImportError(
    "small_network_core is retired and archived at Git tag "
    "archive/small-network-v1. Use `python -m experiments.mnist_conv ...` for "
    "MNIST Conv paper runs."
)
