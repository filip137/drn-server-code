# Four-device intrinsic-reference balance

This validate-only composition tests whether deterministic, weight-blind
identity assignment can make the four intrinsic references in each physical
quad behave like a logical zero.  It compares the sampled identity order with
`reference_balanced_binding_v1` on one development assignment and three
untouched assignments.

The target map is continuous and bounded.  It deliberately makes no decision
about level spacing or level count: every selected branch is represented as
`G=B+d`, and every full `G` enters both transfer and denominator loading.  No
optimizer update, quantization, P&V, HWA, write/read noise, or drift is active.

The balanced assignment is approximate.  It minimizes the signed reference
sum of each quartet; it does not claim that four generic real-valued
references can be made algebraically identical.
