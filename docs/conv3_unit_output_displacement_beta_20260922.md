# Conv3 beta choices for output RMS displacement near one

Recorded September 22, 2026, at Filip's request.

The selected injected betas are **baseline 88.7, ours 1.385, and legacy
0.02173**. Direct replay of the actual saved initialization confirms that
all three give pooled output displacement within **0.031% of one**.

## Target and measured results

The target is `RMS(v_plus - v_free) = 1` at the output layer, in simulator
voltage units. The free state is measured after T iterations; the positive
and negative nudged states follow K additional iterations from that same
free state. RMS is pooled over all output coordinates and all 576 examples,
rather than averaging batch RMS values.

| Scheme | Injected beta | Positive-free RMS | Negative-free RMS | Centered half-difference RMS | Positive deviation from 1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 88.7 | 1.0003009717 | 1.0003009515 | 1.0003009616 | +0.030097% |
| Ours | 1.385 | 0.9998121959 | 0.9998121685 | 0.9998121822 | -0.018780% |
| Legacy | 0.02173 | 0.9998813740 | 0.9998813768 | 0.9998813754 | -0.011863% |

The centered measurement is `RMS((v_plus - v_minus)/2)`. Both nudge signs
and the centered measurement agree to six decimal places. The betas were
estimated from the nearly linear output response in the earlier measurements;
the table above contains new direct replay measurements at those exact betas.

## Replay conditions and interpretation

- Conv3, shared saved seed-0 initialization; 36 identical validation batches
  of 16 ordinary-MNIST examples from the deterministic 55,000/5,000 split.
- T=K=8; float64 centered frozen-current EqProp; explicit perfect diodes;
  input gain 360; wide [0,100] weights; exact-zero biases; clean endpoints.
- Amplification `(voltage,current)`: baseline `(1,1)`, ours `(4,1)`, legacy
  `(4,0.25)`. Values above are **injected beta**; the runner's base beta is
  `injected_beta / (voltage_amp/current_amp)^3`.
- All three full replays and three one-batch smokes validate locally.
  Initial weights and source inputs remain unchanged; all 1,728 recorded
  projected-KKT residual checks pass. No training, accuracy evaluation, or
  official-test access occurred.

These choices achieve the requested output-displacement match for this
initialization and cohort. They do not establish equal hidden-layer
displacements or stability during subsequent clean or noisy training.
The target is a pooled RMS: individual examples and batches need not have
displacement exactly one.

The successful smoke and replay took 199.47 seconds on Akib's RTX 3080.
An initial smoke ran out of GPU memory; its failed bundle is retained. The
successful retry changed allocator/workspace settings while preserving the
scientific configuration, precision and batch size.

## Evidence

- [Detailed report](../results/eqprop-conv3-output-rms1-init-20260922-v1/analysis/report.md)
- [Per-layer RMS measurements and batch ranges](../results/eqprop-conv3-output-rms1-init-20260922-v1/analysis/layer_summary.csv)
- [Coverage validation and source hashes](../results/eqprop-conv3-output-rms1-init-20260922-v1/analysis/validation.json)
- [Replay config](../configs/conv/eqprop_conv3_output_rms1_init_20260922.json)
- [Experimental manifest](experimental_manifest.md#conv3-beta-choices-for-unit-output-displacement-september-22)

The authoritative local run bundles are under
`results/eqprop-conv3-output-rms1-init-20260922-v1/collected/workspace/output-retry/`.
