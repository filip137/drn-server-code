# X-008 — Legacy alignment despite expected attenuation of the nudging signal

## Filip's observation and expectation

After viewing the layerwise cosine-versus-displacement curves, Filip noted on
September 25, 2026 that legacy still appears to have higher cosine similarity
than ours and baseline despite RMS matching. He finds this surprising because
he expected the nudging signal to become smaller and smaller as it propagates
back toward earlier layers, making legacy more vulnerable to read noise.

Record this as an open mechanism question: **why does the expected attenuation
not translate into uniformly worse noisy gradient alignment for legacy under
the chosen output-displacement matching rule?** This question follows the
measurements; it is not a preregistered hypothesis or a new experiment assignment.

Filip's wording: "even with the matching RMS, cos similarity of the legacy runs
is higher than ours and baseline"; this was contrary to his expectation that
"the nudging signal would get smaller and smaller." The discussion below treats
that expectation as attenuation toward earlier layers; the original observation
is preserved separately from this interpretation.

## What the measurements establish

The observation is particularly clear in **Conv3's second and third convolution
layers and its readout**. Legacy's mean cosine exceeds both other schemes at all
25 sampled nonzero-noise/target combinations in each of those layers, with a
margin greater than1e-6. That1e-6 reporting filter only avoids counting numerical
ties; it is a retrospective descriptive threshold, not a significance criterion.
Conv3's first-layer cosines are near zero under the tested noise, and their tiny
scheme differences should not be presented as useful gradient alignment.

It is not a universal layerwise ordering. Ours often leads in Conv1 and Conv2's
convolution layers; the selected worst-layer advantages in those architectures
reproduced across five fresh noise draws. No competitor dominated every layer
simultaneously in the screen. Preserve those findings alongside Filip's observation.

For example, at initial output D/F=2 and endpoint sigma=1e-3, mean cosines against
clean finite-K BPTT are:

| Network / parameter layer | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv1 / convolution1 | 0.845854 | **0.979344** | 0.863110 |
| Conv2 / convolution1 | 0.010019 | **0.017837** | 0.010271 |
| Conv2 / convolution2 | 0.063759 | **0.123440** | 0.089850 |
| Conv3 / convolution1 | -0.009608 | -0.008567 | -0.008644 |
| Conv3 / convolution2 | 0.001046 | -0.000402 | **0.002959** |
| Conv3 / convolution3 | 0.018168 | 0.040412 | **0.093275** |
| Conv3 / readout | 0.998033 | 0.999634 | **0.999996** |

These are means of individual batch/noise-draw cosines, not cosines of averaged
gradients, and not training accuracies. Larger among near-zero values does not
imply reliable credit assignment.

## The matching rule is a material distinction

The completed runs match **normalized output displacement**

`r = D_out/F_out`, where `D = RMS((v_plus-v_minus)/2)` and
`F_out = RMS(post-T free output)`.

They do **not** match absolute output D, the displacement at each hidden layer,
or gradient signal-to-noise ratio. Noise is absolute additive Gaussian noise
on the copied positive/negative noninput endpoint voltages; it is added after
clean physical relaxation. It is not relative noise and does not perturb the
nudging dynamics themselves.

Conv3 at D/F=2 illustrates why attenuation and a legacy cosine advantage can
coexist. The following are measured pooled physical RMS values in simulator
voltage units; hidden1 is nearest the input and hidden3 nearest the output.

| Scheme | Free-output RMS F | Output D | Hidden3 D | Hidden2 D | Hidden1 D | Hidden1 D / output D |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.00600041 | 0.0120008 | 1.27808e-5 | 2.42651e-7 | 5.18425e-8 | 4.320e-6 |
| Ours | 0.0413063 | 0.0826126 | 8.65271e-5 | 5.39837e-7 | 3.75990e-8 | 4.551e-7 |
| Legacy | 0.384026 | 0.768053 | 2.04473e-4 | 9.70946e-7 | 5.19101e-8 | 6.759e-8 |

The legacy signal **does shrink strongly toward the input**, and its first-hidden
/output ratio is smaller than the other two at this point. Yet the matched r
starts legacy from an absolute output displacement approximately64 times baseline
and9.30 times ours. Its absolute hidden2/hidden3 displacements are also larger;
its hidden1 displacement is approximately baseline's and slightly larger than ours.
Thus stronger relative attenuation does not imply a smaller absolute signal in
all layers under this normalization. The chosen normalization is one plausible
contributor to the observed ranking, not a demonstrated complete causal explanation.

Cosine measures direction: multiplying a noiseless gradient by a positive scalar
alone leaves cosine unchanged. In the noisy estimator, direction also depends on
how endpoint noise enters the local energy-gradient calculation, layer voltage
scales and coupling, not just a single output or hidden-voltage RMS. D/sigma is a
voltage-scale diagnostic, not a direct measurement of gradient SNR. Clean/noisy
cosines and gradient-error norms must therefore remain separate controls.

## Open question and next reasoning step

Filip's subsequent working hypothesis on September 25 is that **fixed absolute
read noise favors higher voltages**. At matched r, D=rF, so larger F gives larger
absolute displacement relative to the same noise sigma. This is now registered
as [H-009](../hypotheses/H-009-absolute-noise-voltage-scale.md), motivated by the
measurements above. The [relative-read-noise control](X-004-relative-read-noise.md)
would test the contribution of voltage scale. This explanation is plausible but
has not been isolated experimentally; the existing clean and absolute-noise runs
do not themselves establish its causal contribution to gradient alignment.

The expectation of attenuated nudging should be tested against the measured
**layerwise attenuation ratios**, **absolute hidden displacements**, and
**noise-induced gradient error relative to the clean gradient** together.
Existing state and gradient-control CSVs already contain these quantities; an
initial explanation can use those artifacts without more training or GPU sweeps.
An absolute-D matching control would be a different intervention and is not
already answered by the D/F plots. No such new run is authorized by this note.

Keep the scientific conclusion scoped: legacy remains better aligned in the
specified later Conv3 layers, while ours has confirmed worst-layer advantages
in selected Conv1/2 regions. Why these facts coexist with legacy's training
advantage remains unresolved. Initialization-only cosine does not settle the
training mechanism or imply persistence at trained checkpoints.

## Evidence

- [All layerwise displacement curves](../../../results/conv123-initial-displacement-noise-map-20260925-v1/analysis/displacement_curves/README.md)
  and [combined PDF](../../../results/conv123-initial-displacement-noise-map-20260925-v1/analysis/displacement_curves/all_layer_cosine_displacement.pdf).
- [810 layer means](../../../results/conv123-initial-displacement-noise-map-20260925-v1/analysis/layer_summary.csv),
  [physical RMS controls](../../../results/conv123-initial-displacement-noise-map-20260925-v1/analysis/physical_state_controls.csv),
  [gradient scale/error controls](../../../results/conv123-initial-displacement-noise-map-20260925-v1/analysis/gradient_scale_controls.csv).
- [Final series report](../../../results/conv123-initial-displacement-noise-map-20260925-v1/report.md)
  and [fresh-noise confirmation](../../../results/conv123-initial-displacement-noise-map-20260925-v1/confirmation/analysis/report.md).

All numbers above are reaggregated from the completed seed-0,576-example ordinary
MNIST initialization replay, T=K4/6/8, three screen noise draws and the existing
confirmation. No new measurements, optimizer steps or official-test reads.
