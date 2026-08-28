# IBM OM four-reference identity-balance control

## Question

Can the four-device fixed-reference topology recover a genuine logical zero by
choosing which sampled device identities form each quad, before making any
decision about level spacing or level count?

This is the nonlocal identity-binding upper control in the
[symmetry-reference-anchored study family](ibm_om_reference_anchored_studies.md).

For one quad, the zero-state transfer term is

\[
C_{0,Q}=\frac{r_{++}-r_{+-}-r_{-+}+r_{--}}{2}.
\]

Equal references are sufficient but not necessary.  The actual condition is
`r+++r-- = r+-+r-+`.  Generic real-valued references cannot normally satisfy
this equality exactly through permutation, so this study measures and retains
the residual.

## Frozen intervention

The control keeps the sampled order.  The balanced policy operates separately
within each layer:

1. Map each fitted reference to the circuit coordinate
   `rho=clip((r+1)/2,0,1)` and stable-sort by `(rho, source identity)`.
2. Form consecutive groups of four nearby references.
3. Enumerate the three unique two-versus-two sign partitions and choose the
   one with minimum absolute signed reference-sum mismatch; break ties by the
   lexicographic source-identity tuple.
4. Deterministically shuffle completed quartets over logical quad addresses
   using only assignment seed, layer key, and algorithm version.
5. Move every identity field together.  No identity crosses a layer, and the
   rule reads no weights, labels, bounds, accuracy, or calibration outcome.

This global within-layer regrouping is a physical identity-assignment
intervention.  It is distinct from the shared-zero no-reference Stage 0E
control and does not overwrite that planned experiment.

## Continuous baseline-only mapping

This study deliberately does not choose a level size or a number of levels.
For each selected active branch,

\[
G_i=B_i+d_i,
\]

where `B_i` is the intrinsic reference projected only when the active state
must lie inside that identity's sampled active bounds, and `d_i` is a
continuous nonnegative lift limited by the minimum positive headroom of the
four assigned identities.  Sign is selected by diagonal versus off-diagonal
rail placement.  An unselected reference branch remains at its mapped
intrinsic `r_i`.

No baseline is removed before circuit evaluation.  The solver receives every
full `G_i`; the numerator is derived from the four full conductances and the
denominator retains their full sum.  The run records branch-level `B`, `d`,
and `G`, their hashes, the maximum `G-(B+d)` residual, signed contrast,
loading, headroom, bound checks, sign errors, and voltage diagnostics.

## Calibration and held-out comparison

Assignment 86001 is the only development population.  Both binding policies
search the same 16 layer-scale pairs from `{0.125,0.25,0.5,1.0}^2`, select by
calibration accuracy, calibrated KL, then lexicographic pair, and fit a
positive logit gain.  Both selected calibrations are frozen before evaluating
assignments 87001, 87002, and 87003.

Every held-out population evaluates the full two-by-two binding/calibration
matrix.  The scheme-optimized comparison is `random@cal_random` versus
`balanced@cal_balanced`; the two fixed-calibration comparisons isolate whether
the sign and size of the binding effect depend on calibration.

The metric is `ibm_om.reference_balanced_continuous_init.v1`, a bounded
continuous diagnostic for the baseline decision.  It is not the later
standard-level headline metric because quantization, spacing, and level count
are explicitly absent.

## Exclusions and gate

There is no BPTT/QAT, optimizer update, HWA modifier, deterministic pulse
codebook, P&V controller, stochastic write, inference read noise, retention,
or drift.  This is a normalized fitted-model OM control, not a fabricated
array result or an absolute-Siemens claim.

The accuracy gate is a held-out balanced scheme-optimized mean of at least
90%.  A failure is informative: if reference contrast collapses but accuracy
does not recover, then baseline signed-sum mismatch alone is not the limiting
mechanism.

## Results and interpretation

Both declared native arms completed with 53 artifacts each.  Artifact-verified
coverage and independent semantic replay passed.  On held-out assignments
87001--87003, the ideal bounded continuous results were:

| Binding | Mean accuracy | Range | Exact results |
|---|---:|---:|---|
| Sampled identity order | 84.49% | 80.78--87.80% | 8489, 8780, and 8078 correct out of 10000 |
| Reference-balanced binding | 91.83% | 87.47--94.13% | 9389, 8747, and 9413 correct out of 10000 |

Thus the balanced scheme passed its predeclared 90% mean gate and improved the
scheme-optimized mean by 7.34 percentage points.  With the random-binding
development calibration frozen for both bindings, the paired mean improvement
was 7.68 points, ranging from +0.49 to +13.45 points.  The scheme-optimized
effect was not uniform: assignment 87002 was 0.33 points lower after balancing.

The physical diagnostics strongly support signed reference mismatch as a
failure mechanism.  Depending on assignment and layer, balancing reduced the
intrinsic-reference zero-contrast RMS by 58--150 times.  The nonzero-weight
sign-flip fraction fell from 19.9--20.4% to 0.23--0.29% in the first layer and
from 9.6--11.4% to 0--0.4% in the second layer.

This is not, however, a deployable local initialization method for a fixed
array.  The balancing policy globally regroups complete device identities
within each layer, so it changes the identities, bounds, headroom, and loading
assigned to each quad.  The appropriate closeout outcome is therefore
`inconclusive`: the zero-bias mechanism is supported and the nonlocal upper
control is strong, but the study does not show that a physical array can obtain
the same benefit through local programming.

The next matched study keeps every identity in place and programs a nearby
baseline per conductance.  Its primary intervention is the box-constrained
minimum-L2 adjustment from mapped intrinsic symmetry values subject to exact
quad zero contrast.  It must retain the complete `G=B+d` in the numerator and
denominator and use identical matched `d` tensors across the uncompensated and
compensated arms.  Continuous ideal mapping comes first; the same baseline
objective should then be checked with the standard four-delta codebook and
deterministic pulse distinguishability before introducing stochastic write
noise, HWA, or on-chip recovery.
