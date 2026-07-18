# Legacy Conv diagnostic source snapshot

These files preserve the exact pre-cutover, uncommitted implementations that
occupied the corresponding paths under `experiments/` on 2026-07-18. They are
stored with the non-executable `.legacy` suffix because they import the retired
Conv sweep runner and reconstruct scientific training settings from command-line
arguments.

The active paths are now thin compatibility delegates:

- `create_mnist_bp_conv_tk_preflight.py` emits canonical sweep JSON from a
  complete base run and calibration records;
- `diagnose_mnist_bp_conv_lr_update_ratio.py` consumes validated canonical run
  bundles or a canonical sweep;
- `diagnose_mnist_bp_conv_hardsigmoid_fixed_lr.py` provides the same read-only
  bundle diagnostic restricted to hard sigmoid.

Snapshot SHA-256 values:

```text
2e2159f4c341f9eeda64a47887533b1acb1f7f259e81c96471a22528934db35e  create_mnist_bp_conv_tk_preflight.py.legacy
c44b37bffd0d94747dc722a53a2b388b1f577584f35046642c63cfdc95622c07  diagnose_mnist_bp_conv_hardsigmoid_fixed_lr.py.legacy
e0f18520a51d4be8bd4fa60dc3016840745329641685aa892cca4ea3bb09ca60  diagnose_mnist_bp_conv_lr_update_ratio.py.legacy
```
