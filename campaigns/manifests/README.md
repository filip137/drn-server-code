# MNIST campaign inputs

These manifests are portable templates. They resolve paths relative to this
directory and fail closed when an external input is absent.

Before launching them:

- create the repository-local `.venv/bin/python` interpreter;
- place the measured trace dataset at
  `data/march_slope_x3_5k.hdf5` (expected SHA-256
  `207b143fc1a63710c944db74bc573f16b616ce50e22c6605c12068bf6a3f230d`);
- for `mnist_relu_drn_reset_factorial.json` and
  `mnist_relu_drn_reset_differential_10ep.json`, also place the frozen teacher at
  `data/mnist_relu_teacher_seed42_weights.pt` (expected SHA-256
  `42b0526c4a433057b0ad0f09a7b36ede64afd7d47a3a8985724cac0e46358d54`).

The non-factorial manifests train their teacher as an explicit upstream stage.
Do not replace a frozen factorial input with “the newest” checkpoint: update
the path, digest, and study provenance together when intentionally defining a
new campaign.
