# Four-device ReRAM cohort A-to-B deployment and adaptation

Status: preflight (2026-08-19)

## Question

Does the four-device dual-rail realization retain its cohort-A behavior when
its selected conductances are re-encoded on held-out cohort-B devices using
one common reachable window across every symmetry-related four-cell block?
If immediate deployment loses accuracy, can ten epochs of matched off-chip,
device-constrained fine-tuning recover it?

## Frozen protocol

- Source: selected cohort-A four-device quad-common-window checkpoint from
  `mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1`, SHA-256
  `e95eb276ed30d20c024495480c1f7d985ced853b761b8a3d5f2151aa9614fcb8`.
- Teacher checkpoint SHA-256:
  `42b0526c4a433057b0ad0f09a7b36ede64afd7d47a3a8985724cac0e46358d54`.
- Measured trace SHA-256:
  `207b143fc1a63710c944db74bc573f16b616ce50e22c6605c12068bf6a3f230d`.
- Model: bias-free single-conductance dual-rail `[1568, 100, 20]` DRN,
  `voltage_amp=4`, `current_amp=0.25`, model-local amplification indices.
- Deployment mapping: `dual_rail_quad_common_window`, with `halves` layout on
  the input--hidden matrix and `paired` layout on the hidden--output matrix.
- Device assignment: split seed 42, assignment seed 42, cohort B only, raw
  measured curves, global-nearest projection.
- Deployment control: one zero-learning-rate batch followed by fresh-process
  evaluation on all 10,000 MNIST test examples.
- Adaptation: ten epochs, learning rates `(2.1e-10, 5.7e-13)` inherited from
  the matched cohort-A four-device run, pure `KL(teacher || student)`, and
  validation-KL checkpoint selection. There is no cohort-B learning-rate
  search.

## Formal run contract

- Campaign ID:
  `mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1`.
- Expected stages:
  `cohort_b_quad_deployment_control`,
  `cohort_b_quad_deployment_test`, `cohort_b_quad_train`, and
  `cohort_b_quad_test`.
- Launcher: local subprocess campaign on one CUDA device from a clean Git
  commit. The resolved campaign records the exact source identity and every
  command.
- Raw home:
  `results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/`.
- Expected progress: initialization/deployment metrics before training, then
  one metric record per completed adaptation epoch. Based on the matched
  cohort-A run, completion is expected within approximately 5--15 minutes
  once launched.
- Safe recovery: rerun the identical campaign with `--resume`; completed
  stages may be reused only if the campaign provenance and artifact hashes
  validate. Scientific settings must not change during recovery.

## Results

Pending formal launch and fresh-test validation.
