# Physical-KCL nonlinear updater handoff

Date: 2026-09-02

Branch: `codex/lean-experiment-workflow`

Implementation commit: `acdf930bf7c2a4992a0b182ad3675260c39fad84`

## Outcome

The physical-KCL edge implementation remains the authority. The specialized
nonlinear coordinate updaters now use the same layer factor as the resistive
energy coefficients:

\[
D_l=\left(\frac{B}{A}\right)^{l-1},\qquad
c_l=cD_l,\qquad I_{s,l}=I_sD_l.
\]

Only current-setting diode parameters are scaled. Voltage-domain parameters,
including `V_t`, `V_off`, and `v_off`, remain unchanged.

For `(A,B)=(4,0.25)`, the hidden-layer values are:

| Layer | `D_l` | `c_l` for `c=10` | `I_s,l` for `I_s=1e-6` |
|---:|---:|---:|---:|
| 1 | `1` | `10` | `1e-6` |
| 2 | `1/16` | `0.625` | `6.25e-8` |
| 3 | `1/256` | `0.0390625` | `3.90625e-9` |

## Reconciliation with the earlier SPICE match

At hidden layer `l`, the corrected scalar energy satisfies

\[
\frac{\partial E}{\partial v_l}
=D_l\left(I_{\mathrm{res},l}(v)+I_{\mathrm{diode},l}(v)\right).
\]

The physical-KCL edge correction had already put `D_l` into the quadratic
resistive coefficients `a_l` and `b_l`. The nonlinear energy interactions had
also already put `D_l` into their diode current term. The remaining mismatch
was confined to the specialized coordinate solvers, whose equations still
combined scaled `a_l,b_l` with raw `c` or raw `I_s`.

For example, the quadratic updater was effectively solving

\[
2a_l v+b_l+c f(v)=0
\]

instead of

\[
2a_l v+b_l+cD_l f(v)=0.
\]

After division by `D_l`, the former equation makes the diode `1/D_l` too
strong: `1x`, `16x`, and `256x` at hidden layers 1--3. The exponential updater
had the identical problem with `I_s`.

Legacy inference matched SPICE for a different reason: its coordinate
coefficients encoded the unscaled physical KCL directly and were combined
with the raw diode parameter. Multiplying that complete physical equation by
the positive factor `D_l` does not change its zero. The corrected implementation
therefore keeps the legacy/SPICE equilibrium while also making the coordinate
solver stationary for the corrected scalar energy and its training gradients.

## Implementation

`model/resistive/minimizer.py` now contains one parameter-copy helper that:

1. obtains `l` from the full `Layer_<int>` suffix;
2. computes `(current_amp / voltage_amp) ** (l - 1)`;
3. multiplies only the requested strength key; and
4. returns a copied dictionary, leaving the caller's base configuration and
   all voltage parameters unchanged.

The core `QuadraticMinimizer` applies it per layer to:

- `lpw_diode` and `double_diode_quadratic` via `diode_conductance`;
- `double_diode_exponential` and `single_diode_exponential` via `I_s`.

`labs/custom_minimizer.py` uses the same helper for the custom quadratic path
and every analytical exponential variant selected by its factory, including
float32, experimental-float64, timed, and over-relaxed variants.

This is not double scaling. The energy interaction and the updater are two
separate representations built from independent copies of the raw config:

| Consumer | Raw input | Local value at layer `l` |
|---|---|---|
| nonlinear energy interaction | base `c` or `I_s` | `cD_l` or `I_sD_l` |
| specialized coordinate updater | base `c` or `I_s` | `cD_l` or `I_sD_l` |

The updater never consumes an already-scaled parameter from the energy
interaction.

## Regression coverage

`labs/tests/test_nonlinear_diode_amplification.py` checks:

- the exact factors `1`, `1/16`, and `1/256` at layers 1--3;
- unchanged `V_t`, `V_off`, and `v_off` values;
- no mutation of the caller's parameter dictionaries;
- core LPW, double-quadratic, double-exponential, and single-exponential
  updater construction;
- the custom exponential factory used by the replay; and
- zero total gradient after a quadratic coordinate update at each depth.

Focused suite:

```text
python -m pytest -q \
  labs/tests/test_dense_resistive_amplification.py \
  labs/tests/test_conv_resistive_amplification.py \
  labs/tests/test_hard_sigmoid_amplification.py \
  labs/tests/test_nonlinear_diode_amplification.py \
  labs/tests/test_single_conv.py

132 passed in 1.38s
```

## Exact replay and SPICE verification

The final source was replayed with the same maxicao weights, deterministic
100-input grid, and 256 coordinate-descent sweeps used by the causal diagnostic.
The shared input artifact SHA-256 remained
`d16e1657736b1a43a8f6f59f6e04a9a697b4cd28e0e8002f0c7738cf0a103036`.

| Case | Final source state SHA-256 | Match to runtime diagnostic |
|---|---|---|
| Quadratic | `51928edb73120f51e9bbba9da4d1230f7b552e22e66b8de0011710f0cb07d043` | byte-for-byte |
| Exponential | `90e3bdf82daa9cbce4f7912a69e90420a811ed48a19fdd3bb47f0866b11c801d` | byte-for-byte |

The repository NPZ comparator against the saved SPICE layers reproduced:

| Case | p50 error | p90 error | p99 error |
|---|---:|---:|---:|
| Quadratic | `0.001265%` | `0.001429%` | `0.001561%` |
| Exponential | `0.001193%` | `0.002232%` | `0.006402%` |

The final replay's maximum physical-energy residual was `3.10e-6` for the
quadratic case and `7.58e-5` for the exponential case. Both replay processes
were short local CPU checks (approximately 24 and 32 seconds).

Canonical diagnostic provenance, configs, reference paths, layer-level
results, and the original runtime-only causal test remain in:

```text
/home/filip/server_code/simulation_results/
  amplification_4_current_0p25_crosscheck_20260902/
  PHYSICAL_KCL_BREAKDOWN.md
```

The final implementation replays were written to the temporary verification
root `/tmp/physical_kcl_nonlinear_source_fix_20260902_02`; the durable state
hashes above match the canonical diagnostic artifacts exactly.

## Effect on paper reruns

This correction does not reopen the completed long paper reruns. Those runs
use the active perfect-diode, zero-hidden-bias, no-pooling Conv scope, which
does not invoke these finite quadratic/exponential diode updaters. The closeout
in `docs/legacy_physical_kcl_rerun_inventory.md` remains unchanged.

No production training, remote job, scheduler submission, official-test
evaluation, or long paper rerun was launched for this fix. Only the two
bounded SPICE replay fixtures above were rerun.

## Remaining scope boundaries

- End-to-end SPICE evidence in this handoff validates
  `double_diode_quadratic` and `double_diode_exponential` with fixed scalar
  `(A,B)`. The same necessary strength scaling is applied to LPW and
  single-exponential paths, but those paths do not gain an independent SPICE
  validation from this change.
- Hard-sigmoid already has its own layer-scaled conductance path and is
  unchanged.
- The tabulated experimental I--V updater is unchanged; scaling a measured
  curve requires its own units/normalization decision.
- Pooling interactions, nonzero-bias physical scaling, and dynamic/trainable
  amplification for these non-perfect-diode interactions remain outside this
  validation. In particular, the helper snapshots fixed scalar `A/B` when the
  minimizer is constructed.
- Perfect-diode inference and the already completed physical-KCL paper results
  are unchanged.
