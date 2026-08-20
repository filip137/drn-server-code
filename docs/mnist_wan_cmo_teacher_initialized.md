# Teacher-initialized CMO versus Wan-2022 endpoint comparison

Last updated: 2026-08-20

## Outcome

This study corrects the initialization ambiguity in the earlier CMO/Wan
comparison. Every DRN arm now starts from the same frozen, bias-free
`784 -> 50 -> 10` ReLU teacher. No independently trained DRN checkpoint is
used as the source.

On the untouched 10,000-example MNIST test split, the frozen teacher reached
`97.36%`, and its immediate ideal DRN mapping reached `96.91%`. Two epochs of
generic additive-noise HWA raised the clean DRN to `97.34%`. HWA modestly
improved the first physical write on both devices, while post-deployment noisy
BPTT supplied most of the recovery:

| Test state | CMO accuracy | CMO teacher KL | CMO agreement | Wan accuracy | Wan teacher KL | Wan agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen FP32 ReLU teacher | `97.36%` | `0` | `100.00%` | `97.36%` | `0` | `100.00%` |
| Immediate ideal FP32 DRN mapping | `96.91%` | `0.009345` | `99.00%` | `96.91%` | `0.009345` | `99.00%` |
| Clean HWA DRN | `97.34%` | `0.002681` | `99.52%` | `97.34%` | `0.002681` | `99.52%` |
| Ideal mapping after first endpoint write | `91.15%` | `0.177332` | `92.54%` | `78.62%` | `0.550453` | `79.36%` |
| HWA after first endpoint write | `92.18%` | `0.156257` | `93.51%` | `80.31%` | `0.509778` | `81.06%` |
| HWA + endpoint + selected noisy BPTT | `96.46%` | `0.032341` | `98.11%` | `91.84%` | `0.186471` | `92.62%` |

All KL values are
`KL(teacher || DRN)` after fitting one positive output gain on the fixed
validation cohort. Gain calibration cannot change accuracy. The HWA and BPTT
checkpoints were selected by validation KL, and the test set was evaluated
only after selection.

## Exact initialization

The source checkpoint is a frozen FP32 ReLU teacher with test accuracy
`97.36%`. Its signed matrices are lifted directly into a bias-free,
single-conductance dual-rail DRN with dimensions `[1568, 100, 20]`. Positive
and negative teacher coefficients are placed on their corresponding rails;
there is no DRN optimization before the ideal-map row.

The mapping search selected layer scale fractions `(1.0, 0.125)` in the
shared logical DRN window `0..110 uS` and a positive output gain of
`0.630957`. The immediate mapped checkpoint has SHA-256
`25cdfcbc8fe6dc67e66ab8b847f5ddfe25bd691f0b5be052cec91c0983a45899`.
This mapped checkpoint, rather than the older independently trained
`96.15%` DRN, is the source of both direct device-deployment arms.

The DRN uses `input_gain=100`, `voltage_amp=4`, `current_amp=0.25`, four
asynchronous coordinate iterations, perfect-diode nonlinearities, no learned
bias, and model-local amplification indices. Mapping-scale and output-gain
selection use only the validation data.

## HWA stage

HWA continues from the teacher-mapped checkpoint for two epochs using pure
teacher KL. During each minibatch, temporary independent Gaussian noise is
added to every dense output channel with

```text
sigma_channel = 0.03 * max(abs(channel)).
```

Gradients are computed at that noisy realization; the clean master is then
restored and updated. Validation is noise-free. Epoch 0 was selected by
validation KL (`0.002550`, `97.18%` validation accuracy); epoch 1 was slightly
worse. The selected HWA checkpoint has SHA-256
`e69c10cf560f9082c148ec030c06b5c4141aa1856977a377387caf4d00ab2f18`.
This modifier is a generic IBM-style HWA approximation, not a fit to either
endpoint model.

## Physical endpoint mappings and noise

The finite conductance offsets are retained in the DRN equations. They are
not subtracted on conversion back from the device:

```text
CMO target [uS] = 9 + (88.199997 - 9) * G_DRN / 110 uS
Wan target [uS] = 1 + (40 - 1) * G_DRN / 110 uS

G_effective = G_realized / G_max * 110 uS
```

Consequently, a logical zero becomes an effective `11.22449 uS` CMO edge or
`2.75 uS` Wan edge. Both models use one-day age and device seed 17.
This one-day CMO realization is not interchangeable with earlier one-second
CMO deployment rows.

- CMO samples the AIHWKit CMO/HfOx program-error fit at `0.2%` acceptance,
  followed by the fitted relaxation and read-error terms at one day. The
  result is clamped to `9..88.199997 uS`.
- Wan samples additive Gaussian endpoint error whose conductance-dependent
  standard deviation is the fourth-order one-day Wan-2022 fit, then clamps
  the result to `1..40 uS`. It has no separate read-noise term.

Each endpoint write produces a stored conductance tensor. Noise is not
resampled on every inference. During post-deployment BPTT, however, a fresh
endpoint realization is sampled after every minibatch update.

## Post-deployment BPTT

Both BPTT arms start from the exact selected HWA checkpoint. For every
minibatch, gradients are evaluated at the currently realized conductances, a
clean digital shadow target is updated, and both dense arrays are programmed
again through the corresponding endpoint model. This is closed-loop
program-and-verify modeling, not a Tiki-Taka pulse update.

Ten epochs produced `34,380` rewrite steps. CMO selected epoch 9, reaching
`96.72%` validation accuracy and `0.031577` validation KL. Wan selected epoch
5, reaching `91.36%` and `0.206599`; later Wan epochs had worse validation KL.
Fresh-process test evaluation used the selected checkpoints.

Relative to the HWA first write, noisy BPTT added `+4.28` accuracy points on
CMO and `+11.53` points on Wan. This recovered `82.9%` and `67.7%` of the
respective clean-HWA accuracy gaps. Teacher KL fell by `79.3%` on CMO and
`63.4%` on Wan. HWA alone added only `+1.03` points to the CMO first write and
`+1.69` points to the Wan first write.

## Interpretation

For this device realization, HWA is useful but not the dominant intervention.
The much larger improvement comes from post-deployment BPTT with noisy writes.
This experiment does not yet prove that HWA is necessary for BPTT recovery,
because an ideal-map-to-device-to-BPTT arm without HWA was not run.

The device gap is not ordered by the size of the finite floor. At the HWA
first write, the programming-error RMSE relative to the ideal affine physical
target is `1.152 uS` in effective DRN units for CMO and `5.130 uS` for Wan,
so the tested Wan realization is `4.45x` noisier by this measure. Conversely,
the deterministic affine mapping error relative to the zero-floor digital
target is larger for CMO (`10.681 uS` versus `2.617 uS`) because CMO has the
larger floor. The CMO floor is a highly structured common-mode offset that the
balanced dual-rail input and paired output largely reject; Wan's stochastic,
conductance-dependent error changes relative edge strengths. Total RMSE to
the zero-floor digital target therefore does not predict accuracy by itself.

The large fitted CMO output gain (`19.953`, versus `1.995` for the HWA-Wan
write) also shows that CMO strongly attenuates raw output scores. The retained
decision structure is good enough for accuracy after calibration, but this is
not evidence that analog margin, ADC resolution, or circuit noise is benign.

## Evidence boundary and provenance

This is an exploratory single-teacher, single-device-seed comparison. It does
not model pulse counts, asymmetric potentiation/depression, endurance, energy,
converter quantization, IR drop, dynamic inference noise, or run-to-run
uncertainty. The next controls needed to assess HWA necessity are a matched
no-HWA BPTT arm and multiple endpoint seeds; device-matched HWA should then be
compared with the generic 3% modifier.

- Frozen source commit:
  `f60d4406c5a5cde7b75726626e4adeae5f4e4c03` (clean)
- Teacher SHA-256:
  `42b0526c4a433057b0ad0f09a7b36ede64afd7d47a3a8985724cac0e46358d54`
- CMO selected-BPTT SHA-256:
  `d1224cfb5677df0d943e873b935e090a0b93e3f58f454c4c521e7b56629816ef`
- Wan selected-BPTT SHA-256:
  `4e27f679274aa344331c52b2ca734ae92f773d174b4ffa626db27ea1af1c6ab4`
- Campaign IDs:
  `mnist-teacher-initialized-common-f60d4406-20260820-v1`,
  `mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1`, and
  `mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1`
- Local ignored raw artifacts:
  `results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/`
