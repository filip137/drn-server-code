"""Import-error shim for the archived ``small_network_core`` module."""

raise ImportError(
    "small_network_core is retired and archived at Git tag "
    "archive/small-network-v1. Use `python labs/mnist_train.py ...` for "
    "maintained MNIST Conv training."
)
