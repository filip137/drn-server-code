# Single-rail common-window initialization versus bounded FP32

## Question

Can the existing one-conductance-per-dual-rail-edge MNIST DRN retain a ReLU
teacher initialization when complementary rail conductances are programmed
through a shared reachable window, and how does it compare with a uniform
bounded FP32 network without device-to-device variability?

This is an exploratory, single-seed screen rather than paper-facing evidence.

## Frozen protocol

- Teacher, MNIST split, minibatch order, solver, model-local amplification,
  objective, and 10-epoch budget match the earlier initialized single-device
  study.
- The measured arm uses the same raw cohort-A device split and virtual-device
  assignment (`split_seed=42`, `assignment_seed=42`). It still uses exactly
  one conductance per physical dual-rail edge.
- Complementary output-rail conductances share a pairwise reachable window.
  The nominal encoding is centered inside that window so compressed mappings
  retain symmetric programming margin.
- Candidate layer-scale pairs are `(1, 0.125)`, `(1, 0.25)`, `(1, 0.5)`,
  `(1, 0.75)`, and `(1, 1)`. The measured arm selects on the 1,024-example
  calibration subset after global-nearest measured projection. Test data are
  not used for mapping, gain, epoch, or checkpoint selection.
- The FP32 control has the same single-device topology and centered encoding,
  exact float32 updates, uniform bounds of 70--90 uS, and no device assignment
  or measured-state projection. This range approximates cohort A's 70.7 uS
  pairwise baseline and 18.4 uS mean overlap span.

## Monitoring contract

- Campaign ID:
  `mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1`
- Manifest:
  `campaigns/manifests/mnist_relu_drn_single_pairwise_common_window_vs_fp32_10ep.json`
- Expected stages: measured train/test and bounded-FP32 train/test, four total.
- Target and launcher: local subprocess campaign on the available CUDA GPU.
- Operational output root:
  `/home/filip/server_code/.codex/worktrees/tiki-taka-lora-integration/results/`
- Progress evidence: initialization metric, one metric record per completed
  epoch, selected named weights, epoch-boundary resume state, fresh-process
  test result, and campaign aggregate.
- Expected cadence: initialization within a few minutes and epoch metrics at
  sub-minute cadence; total runtime approximately ten minutes.
- Safe recovery: rerun the identical campaign with `--resume --fail-fast`.
  Completed stages may be reused only when campaign fingerprints and artifact
  hashes validate.

## Result

All four stages completed from clean commit `7e144ace8b573149057bf55d9e3fc63eda20b623`.
An immediate `--resume` replay reused the completed stages after validating
their fingerprints and artifact hashes.

| Arm | Selected layer scales | Initial validation | Epoch-10 validation | Selected checkpoint | Fresh test |
| --- | --- | ---: | ---: | --- | ---: |
| Measured cohort-A common window | `(1.0, 0.5)` | 34.20% | 96.20% | epoch 10 | 96.96% |
| Bounded FP32, 70--90 uS | `(1.0, 1.0)` | 97.26% | 97.44% | initialization | 97.37% |

Selection used validation teacher KL, not accuracy.  The measured arm's
selected epoch-10 checkpoint has validation KL 0.050072 and fresh-test KL
0.041213.  The FP32 initialization exactly preserves the teacher closely
enough that none of its trained checkpoints improves on its initial
validation KL of 7.804e-6; its selected fresh-test KL is 8.180e-6.  The FP32
epoch-10 validation accuracy is reported as an endpoint diagnostic, while its
fresh test uses the protocol-selected initialization checkpoint.

The measured initialization used mean common-window baselines of 70.799 and
71.209 uS and mean spans of 18.323 and 17.934 uS for the two layers.  Empty
pairwise overlaps affected 1.367% and 1.200% of pairs.  Despite small
per-device nearest-state projection errors (0.245 and 0.148 uS RMS), the
independently projected complementary rails accumulated enough mismatch to
reduce initial validation accuracy to 34.20%.

Ten epochs of measured on-chip-style updates recovered 62.00 validation
percentage points and 97.16% of the initial teacher-KL gap.  Its selected
fresh-test accuracy is only 0.41 percentage points below the bounded FP32
control.  Thus the narrow conductance range itself is not the limiting factor:
device-specific projection mismatch damages the initial function, while
analog adaptation recovers nearly all of the classification-accuracy loss in
this single-seed screen.
