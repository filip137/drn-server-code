# Cell-aware exact-bounds IBM OM comparison

These configs implement the exact per-cell affine oracle comparison for
`mnist-ibm-om-cell-aware-exact-bounds-transfer-20260824-v1`.

Every arm starts from `data/ibm_om_cell_aware_full_span_v1.pt` (SHA-256
`a99995a3e5b321a56bb7e00e29840c3b76f3fee95b8d8c80fdf0fa16e93b3563`)
and the frozen ReLU teacher (SHA-256
`9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`).
The OM device model SHA-256 is
`3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3`.

For each assigned cell, the target mapper consumes its exact simulated
logical bounds and intersects them with the DRN's normalized `[0, 1]`
coordinate:

```text
L_i = max(0, (logical_min_i + 1) / 2)
H_i = min(1, (logical_max_i + 1) / 2)
u = (q++ - q+- - q-+ + q--) / 2
n = clamp(round_half_away_from_zero(4u), -4, 4)
T_i = L_i + max(s_i n, 0) (H_i - L_i) / 4
s = (+1, -1, -1, +1)
```

The continuous arm replaces rounded `n` with `4u`. There is deliberately no
cross-cell baseline or span compensation. The mapper never clips, narrows,
reassigns, or substitutes a target after construction. Its immutable
codebook artifact contains the exact bounds, usable floors/spans, logical
shadow values, signed codes/indices, targets, hashes, and sign diagnostics.

Development uses repaired assignment seed `84001`; held-out deployment uses a
fresh repaired assignment seed `85001` only after all four checkpoints are
frozen. Endpoint seeds are `84002` for compact HWA, `84003` for three-repeat
development selection, and `85101` through `85105` for held-out deployment.
The model-local mapped-target gains frozen on assignment `84001` are
`3.5481338923357533` (continuous) and `2.818382931264453` (nine-level).
Their exact population, calibration cohort, grid, target hashes, and
development-only diagnostics are frozen in `cell_aware_design_receipt.json`.

The learning rates are copied exactly from the corresponding arms of the
frozen RESET-relative design receipt; this study performs no learning-rate
selection:

- clean BPTT: `[0.1388429752066116, 0.00037685950413223146]`;
- continuous HWA: `[0.0017531914893617025, 0.000004757787234042554]`;
- quantized QAT: `[0.004411914893617021, 0.000011974808510638298]`.

All trained checkpoints are selected by the highest mean apparent-forward
student accuracy over three fixed development programming repeats. Exact ties
retain the earliest epoch. Held-out outcomes cannot change a gain, learning
rate, codebook mode, checkpoint, or assignment.
