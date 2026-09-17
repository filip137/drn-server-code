# RESET-relative deployed-array recovery pilot

These configs implement the one-seed protocol in
[`docs/ibm_om_reset_relative_on_chip_recovery.md`](../../../docs/ibm_om_reset_relative_on_chip_recovery.md)
and belong only to
`mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1`.

`development_deployment.json` creates the one immutable source deployment on
assignment `84501`, endpoint seed `84601`. The remaining twelve configs cross
four update paths with slow-pulse budgets `0.1`, `1`, and `4` per slow cell:

- `rail_refresh_*`: gradient-free matched-rail pulses;
- `direct_pulse_*`: individual-conductance stochastic SET/RESET;
- `tiki_taka_ideal_*`: ideal continuous fast accumulation and cyclic transfer;
- `tiki_taka_physical_*`: commissioned repaired-OM fast pairs and physical
  transfer/reset.

Prepare the study without launching it:

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1.json \
  --results-root results
```

Create the source deployment first:

```bash
python -m ebl validate \
  --config examples/mnist_relu_drn/ibm_om_reset_relative_on_chip_recovery/development_deployment.json \
  --weights <frozen-reset-relative-qat-weights.pt> \
  --teacher-weights <frozen-relu-teacher.pt> \
  --device-model data/ibm_reram_om_pv128_hwa_v1.json \
  --output-dir results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/deploy-source-development
```

Every recovery invocation receives that run's exact deployment sidecar:

```bash
python -m ebl train \
  --config examples/mnist_relu_drn/ibm_om_reset_relative_on_chip_recovery/direct_pulse_budget_1.json \
  --weights <frozen-reset-relative-qat-weights.pt> \
  --deployment <source-run/artifacts/ibm_om_deployment.pt> \
  --teacher-weights <frozen-relu-teacher.pt> \
  --device-model data/ibm_reram_om_pv128_hwa_v1.json \
  --output-dir results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/direct-pulse-budget-1
```

Set `EBL_AIHWKIT_PYTHON` to the pinned AIHWKit 1.1.0 interpreter for the
physical-fast arms. Do not add seeds, select an intermediate epoch, or use a
test split in this pilot.

