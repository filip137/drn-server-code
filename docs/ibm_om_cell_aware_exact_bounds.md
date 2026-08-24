# IBM OM cell-aware exact-bounds quantization

## Question

This study asks how the RESET-relative transfer result changes when target
generation is allowed to retain exact characterized values for every physical
cell. It is an oracle information ablation: deployment on assignment `85001`
receives that assignment's fresh simulated cell bounds. The result therefore
measures the benefit and failure mode of cell-specific affine placement; it
does not claim a no-characterization transfer method.

## Target construction

The clean optimizer owns continuous FP32 conductances. For one canonical
four-cell quad, normalize them to `q` and reconstruct

```text
u = (q++ - q+- - q-+ + q--) / 2.
```

For each assigned cell `i`, form its usable exact interval

```text
L_i = max(0, (logical_min_i + 1) / 2)
H_i = min(1, (logical_max_i + 1) / 2).
```

The quantized mode uses nine signed codes

```text
n = clamp(round_half_away_from_zero(4u), -4, 4),
```

while the continuous mode uses `n = 4u`. With rail signs
`(+1,-1,-1,+1)`, every target is

```text
T_i = L_i + max(s_i n, 0) (H_i - L_i) / 4.
```

There is no cross-cell baseline cancellation or span compensation. A zero
code sends all cells to their own lower bounds and can therefore produce a
nonzero logical contrast. That effect and any nominal logical sign flip are
reported rather than repaired.

Quantization precedes endpoint sampling and exact program-and-verify. Neither
the nominal targets nor programmed endpoints are rounded afterward. Because
the nominal target is constructed inside each usable range, compact HWA has no
out-of-bound fallback path for repaired arrays. Physical programming failure,
acceptance noise, and cap-128 exhaustion remain observable outcomes.

## Matched design

All scientific inputs match the shared RESET-relative study: full-span shadow,
teacher, DRN topology, data split, repaired assignments `84001` and `85001`,
endpoint seeds, device model, checkpoint-selection rule, and formal coverage.
The corresponding RESET-selected learning-rate schedules are reused verbatim;
cell-aware results do not select another rate. Assignment `85001` remains
embargoed until the four development checkpoints are frozen.

The four development pipelines are zero-update quantized initialization,
clean BPTT followed by quantized deployment, continuous cell-aware HWA, and
nine-level cell-aware QAT. Selection maximizes the mean apparent-forward
validation accuracy over three fixed development programming repeats, with
the earliest epoch retained on exact ties. Each selected checkpoint receives
five pulse-resolved held-out deployments, using endpoint seeds `85101` through
`85105`.

## Artifact contract

Every physical selection or deployment saves a separate immutable codebook
artifact with:

- population fingerprint, assignment, binding order, shapes, and canonical
  halves/paired layouts;
- exact logical bounds, their normalized exact coordinates, and the usable
  floors, ceilings, and spans;
- the source conductances, reconstructed logical shadows, continuous code
  coordinates, and signed code/index tensors for quantized operation;
- requested targets and exact tensor hashes;
- cell-specific baseline contrasts, nominal mapped contrasts, sign vectors,
  and sign-flip diagnostics; and
- an explicit declaration that hidden bounds were consumed by target
  generation.

The deployment sidecar retains the same codebook alongside apparent and
persistent endpoints, acceptance/failure masks, pulse costs, population state,
and RNG continuation. The read-only decomposition rebuilds both targets and
codebook bit-exactly before evaluating any counterfactual state.

## Interpretation boundary

The primary hypothesis compares quantized and continuous cell-aware HWA on the
held-out array. A later corresponding-arm comparison with RESET-relative
initialization is allowed only after exact provenance and coverage checks.
This study stops before on-chip updates and cannot establish that Tiki-Taka,
LoRA, or soft-pulse fine-tuning is required.
