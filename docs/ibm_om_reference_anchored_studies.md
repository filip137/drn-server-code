# IBM OM symmetry-reference-anchored memristor studies

This is the umbrella for IBM OM schemes whose stored zero-state conductance is
derived from each device identity's intrinsic fitted symmetry reference
`r_i`.  The short name is **reference-anchored studies**.

Reference anchoring is independent of physical topology:

- a four-device rail quad can place cells at `r_i`, or near `r_i`, without a
  separate reference branch on every edge; and
- an eight-device topology can use an explicit active/reference pair on every
  edge.

It is therefore more precise than using “reference memristor” as though every
member had an extra physical reference device.  Exact fixed-reference schemes
use `B_i=r_i`.  Local compensation keeps intrinsic `r_i` immutable but stores
a nearby baseline `B_i≈r_i` chosen to make the four-cell logical zero exact.

Every family member obeys the same circuit invariant:

```text
G_i = B_i + d_i.
```

The complete nonnegative `G_i` enters both signed transfer and denominator
loading.  Reference terms may cancel algebraically from transfer; they are
never removed from physical loading.

## Current studies

1. The fixed-reference ideal screens establish that a homogeneous reference
   is not intrinsically inaccurate.
2. The four-reference identity-balance study is a nonlocal upper control: it
   greatly reduces signed reference mismatch, but requires physical identity
   regrouping.
3. The fixed-binding local-compensation study leaves identities and intrinsic
   `r_i` unchanged and moves only the stored baseline by the smallest bounded
   amount needed for exact quad zero.

## On-chip-learning hypothesis

A reference-anchored state may help on-chip learning for two separate reasons:

- starting near the device symmetry point can provide more balanced
  potentiation/depression headroom and reduce systematic update bias; and
- a stable reference can preserve a physically meaningful zero while the
  active conductance is updated.

“Centered” has three different meanings that must be recorded separately:

- the range midpoint balances available upward and downward headroom;
- the pulse symmetry point balances expected SET and RESET step sizes; and
- the locally compensated `B_i` enforces circuit zero and need not equal
  either of the first two points.

The sampled OM `r_i` approximates the fitted pulse symmetry point; it is not
proof of the range midpoint or of an exactly programmable quad zero.

Neither benefit has been demonstrated by the ideal initialization studies.
The local compensated baseline is only near `r_i`, and many required trims are
smaller than one nominal OM pulse increment.  A fixed reference also adds
denominator loading, reference mismatch and read noise; a sign change in the
four-device sign-selected construction requires an explicit role-exchange
rule.  If both members of an active/reference pair are trained, the second
member is no longer a fixed reference and must be treated as a distinct
push-pull update scheme.

The smallest valid learning test must first establish a realizable discrete
baseline and persistent P&V codebook.  It should then start every arm from the
same named persistent deployment and compare frozen deployment with matched
on-chip updates.  To isolate whether centering helps learning rather than
merely initialization, use a two-by-two factorial:

| Baseline placement | Frozen control | Persistent-pulse recovery |
|---|---|---|
| Shared-zero lower placement | `L0` | `L1` |
| Matched centered placement | `C0` | `C1` |

The learning-specific effect is the difference in recovery gains,

```text
(accuracy(C1,end)-accuracy(C0,start))
 - (accuracy(L1,end)-accuracy(L0,start)).
```

All four arms must share topology, identities, deployed states, minibatches,
update budget and RNG policy.  Frozen controls execute the same forward and
backward work but issue no pulses.  Report persistent transition bias,
saturation, return-to-zero error, pulse count, common-mode drift, zero
contrast, loading, voltage, endurance and accuracy.  Tiki-Taka or LoRA is
introduced only as a predeclared matched recovery arm, not assumed necessary
from initialization accuracy alone.
