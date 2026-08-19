# Four-device ReRAM cohort A-to-B deployment and adaptation

Status: complete (2026-08-19)

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

| Stage | Test accuracy | Teacher agreement | Teacher KL |
|---|---:|---:|---:|
| Cohort A, selected four-device source | 97.46% | 98.68% | 0.0183103 |
| Cohort B, immediate quad-common-window deployment | 35.79% | 35.72% | 1.761729 |
| Cohort B, after ten adaptation epochs | **97.31%** | **97.61%** | **0.0557048** |

The immediate deployment loses `61.67` percentage points relative to the
cohort-A source. Ten matched off-chip epochs recover `61.52` points, or
`99.76%` of that classification loss, and reduce test KL by `96.84%`. The
adapted checkpoint remains `0.15` points below its cohort-A source. The
selected checkpoint is the tenth completed epoch (zero-indexed epoch 9); its
validation accuracy, agreement, and KL are `97.14%`, `97.58%`, and
`0.0611194`. The deployment-control and adaptation arms have identical
initial cohort-B assignments and metrics (`35.68%` validation accuracy,
`35.86%` agreement, and `1.792735` KL), confirming deterministic
initialization before their update rules diverge.

For the matched eight-device paired-common-window experiment, immediate and
adapted cohort-B test accuracies were `42.96%` and `97.81%`, with KL values
`2.03767` and `0.03156`. Thus four devices are `7.17` accuracy points below
eight devices immediately after reassignment and `0.50` points below after
adaptation. The immediate four-device KL is nevertheless `13.54%` lower,
showing that classification and teacher-logit fidelity need not rank the two
initializations identically. After adaptation, four-device KL is `76.50%`
higher than eight-device KL.

## Reachable-window and signed-drive diagnostics

| Cohort | Layer | Empty four-cell windows | Mean span | Projection RMS |
|---|---|---:|---:|---:|
| A | input--hidden | 5.837% | 11.145 uS | 0.350 uS |
| A | hidden--output | 6.000% | 11.244 uS | 0.272 uS |
| B | input--hidden | 4.936% | 11.292 uS | 0.319 uS |
| B | hidden--output | 5.200% | 10.519 uS | 0.308 uS |

Cohort B is not failing because its four-cell intersections are uniformly
narrower: its window spans and empty-window rates are close to those of the
successful cohort-A source, and no nominal target is clipped. The problem is
that every logical weight is re-encoded through a different collection of
four measured curves. If

```text
a = g++ - g-+
b = g+- - g--
signed drive = (a - b) / 2
common residual = (a + b) / 2,
```

then the layerwise signed-drive RMS contracts from `3.401` and `4.023 uS` in
the cohort-A checkpoint to `0.552` and `0.550 uS` immediately on cohort B,
factors of `6.16` and `7.32`. Meanwhile, the mean four-cell load rises from
`308.011` and `310.649 uS` to `337.222` and `338.116 uS` (`9.48%` and
`8.84%`). A scalar readout gain cannot undo edge-specific signed rescaling,
symmetry residuals, and changed passive denominators.

After adaptation, signed-drive RMS becomes `2.822` and `0.662 uS`. It does
not simply restore the cohort-A weight matrix: the effective signed-weight
cosines against the source are `0.472` and `0.592`. Instead, joint
device-constrained optimization finds a different cohort-B-specific solution
that compensates across layers while remaining on measured conductance
states.

## Interpretation

The four-device common-window representation remains expressive enough for a
high-accuracy solution on cohort B, despite having one conductance for each
collapsed pair sum and therefore less range and mismatch averaging than the
eight-device construction. It is not initialization-portable across device
cohorts. Differential initialization cancels the nominal common baseline,
but it does not make independently projected devices symmetric, preserve the
source signed scale, or preserve the passive load. The eight-device common-
window arm is more robust before adaptation and retains better logit fidelity
after adaptation; the four-device arm recovers almost all classification
performance while using half as many conductances per teacher weight.

This is a deterministic, single-seed comparison. It establishes feasibility
for this measured cohort and training budget, not universal robustness or an
in-situ programming result.

## Integrity and reproducibility

- All four campaign stages exited successfully from clean implementation
  commit `8325b2a2d503a0b1afe49ecb052bf2bfe3fd863a`.
- The adaptation stage contains exactly ten completed epoch records, an
  epoch-10 resume checkpoint, finite metrics throughout, and a selected
  tenth-epoch checkpoint. Both evaluations used a fresh process and all
  10,000 MNIST test examples.
- An immediate clean-worktree `--resume` audit validated stage fingerprints
  and result hashes and reused all four stages without creating an
  `attempt-002` directory.
- The selected cohort-B weights have SHA-256
  `18da2a946e251749c6f3d51ecbe3930ec2181e6f80db21bbb7a53f58ef9510b0`.
  The zero-update deployment weights have SHA-256
  `a48e78609db1aaf25ea45289a883e2c342ad0ea697dba3609d894932a0c8ed7f`.
- Raw ignored artifacts:
  `results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/`.
- Versioned campaign manifest:
  `campaigns/manifests/mnist_relu_drn_single_quad_common_window_cohort_b_10ep.json`.
