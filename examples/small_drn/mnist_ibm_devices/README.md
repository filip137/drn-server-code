# MNIST IBM endpoint-device configs

These configs implement the study in
[`docs/mnist_ibm_pcm_cmo_noisy_recovery.md`](../../../docs/mnist_ibm_pcm_cmo_noisy_recovery.md).

CMO mapping is an explicit part of the experiment:

- `affine_floor` maps logical `[0,1]` across physical
  `[9,88.199997] µS` and retains the floor in the effective DRN while
  preserving learned-value ordering;
- `literal_conductance` maps `1.0` to `88.199997 µS` and clips every target
  below `9 µS`, making it an absolute-scale/hard-clipping stress test;
- `normalized_offset` uses affine physical targets but subtracts and rescales
  the floor, so it requires an explicit differential/reference-cancellation
  assumption.

The archived 2026-07-29 matched CMO runs used `normalized_offset`; their
resolved configs and exact source bundle remain under
`results/mnist-ibm-pcm-cmo-noisy-recovery/`.

Given an existing FP32 named-weights artifact:

```bash
python -m ebl train \
  --config examples/small_drn/mnist_ibm_devices/hwa_from_fp32.json \
  --output-dir results/<study>/hwa \
  --weights <fp32-weights.pt>
```

Use the selected HWA `checkpoints/weights.pt` for both device arms. A
deployment probe is invoked with `--weights`; its device-deployment metric is
`metrics.initial_validation`, measured before the deliberately zero-rate
single probe update:

```bash
python -m ebl train \
  --config examples/small_drn/mnist_ibm_devices/pcm_deployment_probe.json \
  --output-dir results/<study>/fp32-pcm-probe \
  --weights <fp32-weights.pt>

python -m ebl train \
  --config examples/small_drn/mnist_ibm_devices/cmo_deployment_probe.json \
  --output-dir results/<study>/hwa-cmo-probe \
  --weights <hwa-weights.pt>

python -m ebl train \
  --config examples/small_drn/mnist_ibm_devices/cmo_affine_floor_deployment_probe.json \
  --output-dir results/<study>/hwa-cmo-affine-probe \
  --weights <hwa-weights.pt>
```

For the controlled seed-17 affine run, `metrics.initial_validation` is
`96.12%`. The zero-rate probe performs a second noisy write; its
`last_validation` and stored selected checkpoint are `96.11%`.

Full recovery loads the complete HWA checkpoint:

```bash
python -m ebl train \
  --config examples/small_drn/mnist_ibm_devices/pcm_full_bptt_recovery.json \
  --output-dir results/<study>/pcm-full \
  --weights <hwa-weights.pt>
```

LoRA recovery loads only its frozen base:

```bash
python -m ebl train \
  --config examples/small_drn/mnist_ibm_devices/pcm_lora_bptt_recovery.json \
  --output-dir results/<study>/pcm-lora \
  --base-weights <hwa-weights.pt>
```

Replace `pcm` with `cmo` for the second device. Every full-recovery W1/W2
update and every LoRA-factor update is followed by a fresh endpoint
program-and-verify realization.

The coordinate-update mechanism and floor-mitigation ablations are documented
in
[`docs/conductance_floor_mitigation.md`](../../../docs/conductance_floor_mitigation.md).
