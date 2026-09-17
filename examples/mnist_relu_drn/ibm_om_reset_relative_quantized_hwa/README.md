# Shared RESET-relative four-device quantization study

These configs implement the four-device initialization in
[`docs/ibm_om_reset_relative_quantization.md`](../../../docs/ibm_om_reset_relative_quantization.md).
They belong only to
`mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1`.

## Frozen logical input

Every arm starts from
`data/ibm_om_reset_relative_full_span_v1.pt` (SHA-256
`e9a603833a607aa6fd9060781a4ba9353e0bcf9a04265ef5afb118146b435acb`).
It is a zero-update expansion of the selected lower-placed fractions
`(0.25, 0.125)` in the frozen bounded checkpoint to `(1, 1)`. The source
checkpoint SHA-256 is
`0865b16dbf504a902f42914c719ec1685e80189957b97a77d101efd85efbc4fb`,
and the derivation receipt SHA-256 is
`5ff1d3a4e55e14f3220adf63f522f66ad460487bd8f5117185507916ae5ccd8d`.
The frozen ReLU teacher SHA-256 is
`9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`.

## Mapper

Commissioning performs eight RESET-pulse/read observations on every cell.
For each physical four-cell logical group, its shared baseline is the largest
of the four observed RESET means after adding three estimated standard errors.
Only this measured baseline is array-specific. The target offsets are the
same global codebook on every array:

```text
delta = 0.095849
signed contrast = n * delta, n in {-4, ..., 4}
cell offset = |n| * delta / 2 on the sign-selected diagonal
```

The two inactive cells remain at the shared quad baseline. The target mapper
does not read a sampled cell's hidden minimum or maximum. Those hidden bounds
are consulted only after target construction for an audit and to route the
small non-corrupt out-of-support subset through the exact pulse controller
instead of an unavailable compact surrogate class.

The quantizer reconstructs the signed logical value `u` from the four clean
FP32 cells and applies half-away-from-zero rounding to `clamp(4*u, -4, 4)`.
Quantization therefore occurs before compact endpoint sampling or exact
program-and-verify. During QAT, the mapped endpoint is used for the forward
pass while gradients update the restored FP32 master weights.

The continuous control uses the identical commissioning, span, array,
programmer, and maximum contrast but keeps `4*u` continuous. Its fitted
model-local forward gain is `707.945784384138`; the quantized gain is
`562.341325190349`.

## Declared arms

- `zero_update_quantized.json`: no optimizer updates; quantized initialization
  and development-array deployment only.
- `clean_bptt_quantized_deploy.json`: clean FP32 BPTT, selected and deployed
  through the quantized physical mapper.
- `continuous_hwa.json`: compact endpoint HWA with continuous RESET-relative
  contrasts.
- `quantized_qat.json`: compact endpoint HWA with the nine-level quantizer in
  every minibatch forward.
- `heldout_{quantized,continuous}_seed_85101.json` through seed `85105`:
  pulse-resolved validation on the untouched assignment seed `85001`.

Development HWA uses assignment seed `84001`, compact endpoint seed `84002`,
and pulse-resolved selection seed `84003`. The held-out assignment must not be
sampled until all four development checkpoints are frozen. Each trained arm
is selected by the highest mean apparent-forward student accuracy across
three physical repeats, with the earliest epoch winning an exact tie.

## Predeclared design screens

The RESET-read count was selected before the formal study using assignment
seed `84001`. Four reads put four targets outside the public `[0, 1]`
coordinate and were rejected. Eight reads were the smallest candidate that
kept all targets in `[0, 1]`; its quantized mapping placed 99.2393% of quads
fully inside their hidden supports and the exact baseline replay programmed
99.2456% of cells within 128 pulses. Hidden support was an acceptance audit,
not an input to the targets.

Learning rates were frozen from a non-formal 256-minibatch development-array
screen. The selected multipliers were `8x` for clean BPTT, `16x` for
continuous HWA, and `32x` for quantized QAT; the quantized choice was the
largest screened multiplier retaining at least 97% clean validation accuracy
on the 400-example screen. These results are design provenance, not study
evidence, and the held-out seed was not inspected.

This study stops after off-chip training and deployment. It does not contain
Tiki-Taka, LoRA, or soft SET/RESET recovery and cannot establish that on-chip
training is required.

## Workflow entry points

The ignored full-span prerequisite is reproduced, only at fresh output paths,
with:

```bash
python -m experiments.mnist_relu_drn.ibm_om_reset_relative_initialization \
  --source results/mnist-relu-to-bounded-fp32-map-20260824-v1/runs/bounded-map/20260824T134046.646663Z-d07d32ea-498b4429/checkpoints/weights.pt \
  --output data/ibm_om_reset_relative_full_span_v1.pt \
  --receipt data/ibm_om_reset_relative_full_span_v1.receipt.json \
  --expected-source-sha256 0865b16dbf504a902f42914c719ec1685e80189957b97a77d101efd85efbc4fb \
  --expected-teacher-sha256 9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52
```

Prepare, but do not thereby launch, the tracked study with:

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1.json \
  --results-root results
```

After a completed run has produced its pulse-resolved sidecar, the read-only
decomposition supports this mapper and replays targets from the saved RESET
commissioning observations:

```bash
python -m experiments.mnist_relu_drn.ibm_om_deployment_decomposition \
  --config <training-config.json> \
  --weights <run/checkpoints/weights.pt> \
  --teacher-weights <teacher.pt> \
  --deployment <run/artifacts/ibm_om_deployment.pt> \
  --output <fresh-analysis.json>
```

Do not invoke any `heldout_*` config as a development canary. Their first
native construction samples assignment seed `85001` and ends the declared
assignment embargo.
