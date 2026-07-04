# Fixed-Step Conv DRN Redo Run Plan

Updated: 2026-07-03

## Purpose

Redo the focused Conv2/Conv3 MNIST BP/DRN amplification runs after the
`adaptive_equilibrium=True` default was found in historical configs.

Use only well-performing BP/DRN Conv2/Conv3 runs as LR references. Do not use
Hopfield EqProp, feed-forward, or fully connected controls to choose these LRs.
Those rows may inform architecture intuition, but not this redo's DRN LR
settings.

The focused amplification set is:

- `mnist_bp_amp_v1_c1`: no-amplification baseline.
- `mnist_bp_amp_v4_c1`: voltage-amplified row that performed best historically.
- `mnist_bp_amp_v4_c0p25`: legacy high-voltage/low-current diagnostic row.

`v4/c0.25` remains legacy/diagnostic unless the paper protocol is explicitly
changed.

## Geometry And Required Config

Use padding `1` and the requested stride geometry:

| Architecture | Channels | Kernel | Padding | Strides | Hidden shapes |
|---|---:|---:|---|---|---|
| Conv2 | `64,128` | `3` | `[1,1]` | `[2,2]` | `[64,14,14]`, `[128,7,7]` |
| Conv3 | `64,128,256` | `3` | `[1,1,1]` | `[2,2,1]` | `[64,14,14]`, `[128,7,7]`, `[256,7,7]` |

Implementation requirement:

- Add per-layer `--strides` and `--paddings` support to the BP conv train,
  calibration, and diagnostic entrypoints, while keeping scalar `--stride` and
  `--padding` backward-compatible.
- Every new source config must save the exact `conv_pipeline`.
- Every new run must pass an explicit minimizer config equivalent to
  `labs/configs/mnist_minimizer_fixed_iterations.json`.
- `model_base.minimizer.adaptive_equilibrium` must be `false`.

If `T != K` is selected by diagnostics, add separate CLI/config support for
`num_iterations_inference` and `num_iterations_training` before final runs.

## DRN Reference Settings

These rows are LR/gain references only. They are not automatically paper-valid
because many historical configs omitted the explicit minimizer block.

| Block | DRN reference | Reference setting | Result |
|---|---|---|---|
| Conv2 hard sigmoid | sat30 fixed-amplification 50-epoch rows | `v1/c1`: gain `221.068`, LR mult `4`, LR `0.0260553`; `v4/c1`: gain `158.035`, LR mult `0.5`, LR `0.00455595`; `v4/c0.25`: gain `140.196`, LR mult `0.25`, LR `0.00256783` | best/final: `96.72/96.72`, `97.63/97.59`, `91.90/91.83` |
| Conv2 perfect diode | 50-epoch fixed-amplification rows | shared gain `100`, LR `0.012`, `K=4` | `v4/c1` best/final `98.66/98.64`; `v2/c1` `98.35/98.35` |
| Conv3 hard sigmoid | pad1 DRN saturation/LR runs | best known `v4/c1`: target sat10, gain `296.905`, LR mult `2`, LR `0.00970008`, `K=8` | `96.71/96.64` |
| Conv3 hard sigmoid legacy | low-gain `v4/c0.25` diagnostic | gain `200`, LR mult `0.5`, LR `0.0036`, `K=8` | `83.77/83.77`; diagnostic only |
| Conv3 perfect diode | input-gain DRN screen | best known gain `360`, LR `0.0216`, `K=8` | `v4/c1` 10-epoch `94.37/94.37`; final LR rows not locally available in current state |

For Conv3 hard-sigmoid `v1/c1`, first use the existing DRN sat10 LR-screen
artifacts if they can be synced or read. If that selected row is unavailable,
use LR mult `2` as the architecture-level fallback because it is the explicit
good Conv3 pad1 DRN LR multiplier. Do not run a new broad LR screen just to
resolve this.

## Redo Run Sequence

1. Patch/dry-run geometry and minimizer provenance.
2. Run fixed-step solver diagnostics before final accuracy:
   - `T={6,16,24,48,96,256}`;
   - `K={1,2,4,6,8,16,32,64,128,256}`;
   - record residuals, EP/BP cosine at beta `0.25`, and per-parameter gradient
     norms/zero fractions for early conv weights.
3. Recalibrate hard-sigmoid target sat30 gains for the new geometry and each
   selected amplification row.
4. Transfer DRN LR multipliers, not raw old LRs, to the recalibrated
   hard-sigmoid gains:
   - Conv2: `v1/c1` x`4`, `v4/c1` x`0.5`, `v4/c0.25` x`0.25`.
   - Conv3: `v4/c1` x`2`, `v4/c0.25` x`0.5`, `v1/c1` from synced DRN
     selected-LR artifact or fallback x`2`.
   - Compute LR as `1.44 / calibrated_gain * lr_multiplier`.
5. Transfer perfect-diode DRN references directly:
   - Conv2: shared gain `100`, LR `0.012`.
   - Conv3: shared gain `360`, LR `0.0216`.
6. Run one seed-0 long check per final row using the transferred settings.
7. If seed-0 checks are finite and learn normally, run final seeds `0,1,2` with
   one common epoch budget per architecture/nonlinearity block.
8. Run final saturation and gradient diagnostics on both `best_model.pt` and
   final checkpoints.

## What Not To Redo

- Do not run broad LR screens when the DRN references above already provide a
  usable LR multiplier or LR.
- Do not use Hopfield EqProp, feed-forward, or FC controls as LR references.
- Do not promote old adaptive-equilibrium rows to final evidence.
- Do not tune `v4/c0.25` separately into a main-result row; keep it diagnostic
  unless the protocol changes.

## Acceptance Checks

A redo row can enter the clean comparison only if:

- saved `conv_pipeline` matches the requested geometry;
- saved minimizer config is explicit and has `adaptive_equilibrium=false`;
- hard-sigmoid gain came from the fixed sat30 calibration for that exact
  geometry;
- LR was transferred from the DRN reference rule above;
- fixed-step T/K diagnostics passed and were archived with the run.
