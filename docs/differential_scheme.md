# Differential memristor scheme

The simulated differential scheme replaces each nonnegative DRN edge
conductance `G` with two nonnegative devices, `G+` and `G-`. In unit-gain
energy coordinates, its edge energy is

```text
0.5 G+ (v_pre - v_post)^2 + 0.5 G- (v_pre + v_post)^2.
```

After expansion, signal transfer depends on `G+ - G-`, whereas node loading
depends on `G+ + G-`. If both devices are programmed around the same baseline
`B_floor`, so that `G+ = B_floor + delta+` and
`G- = B_floor + delta-`, the represented signed transfer is
`delta+ - delta-`: the common floor cancels from the cross term. The floor does
**not** cancel from electrical loading, device current, power, or the DRN
conductance-sum denominator.

The displayed unit-gain energy is the abstract signed-branch primitive. With
arbitrary finite positive voltage and returned-current magnitudes `A` and `B`,
the raw physical KCL is nonreciprocal and must not be identified directly with
the gradient of this expression in unscaled node voltages. For an interior
edge, the active-port realization instead uses

```text
i_pre  = A B (G+ + G-) v_pre - B (G+ - G-) v_post
i_post =     (G+ + G-) v_post - A (G+ - G-) v_pre.
```

It has a weighted energy when the dynamic-layer metrics obey

```text
m_1 = 1
m_(l+1) = m_l B / A,
```

and the edge from layer `l` to layer `l+1` contributes

```text
0.5 m_(l+1) [
    G+ (A v_pre - v_post)^2
  + G- (A v_pre + v_post)^2
].
```

The gradient at each dynamic layer is `m_l` times its physical KCL residual.
Thus `A B = 1` is not needed for integrability; it only makes the ideal
source-side and destination-side self-loading coefficients equal. The clamped
input boundary is unity gain and contributes

```text
0.5 m_1 [G+ (v_input - v_1)^2 + G- (v_input + v_1)^2],
m_1 = 1.
```

For constrained ideal-diode nodes, equilibrium means the physical KCL
residual satisfies the voltage-set KKT condition, not necessarily that every
raw current is zero. Because every `m_l` is positive, the physical and weighted
energy KKT conditions are equivalent. Numerical validation should divide the
energy gradient by `m_l`, or use a projected physical KKT residual, before
comparing layers.

## Measured-device initialization

For two measured devices with ranges `[lo+, hi+]` and `[lo-, hi-]`, the
initialization uses their shared reachable window:

```text
common_lo = max(lo+, lo-)
common_hi = min(hi+, hi-)
```

Both devices first start at RESET pulse zero and are then written once. A zero
weight targets `G+ = G- = common_lo`; a positive weight raises `G+` within the
common window, and a negative weight raises `G-`. Each requested value is
projected to the globally nearest measured pulse-conductance state, so the
remaining zero error is the much smaller difference between the two projection
errors. Pairs without an overlapping window cannot encode a nonzero weight in
this mapping.

## Physical interpretation

The two devices are not ordinary parallel memristors: parallel devices would
add their conductances. The `G-` branch needs the opposite voltage polarity,
and the active two-port law needs a signed current returned to the
pre-synaptic dynamic node. A physical realization therefore requires explicit
output-current sensing and signed current feedback, or an equivalent active
two-port; an ordinary pair of positive voltage buffers is insufficient. The
resulting raw two-port is generally nonreciprocal even though its positive
layer-weighted KCL field is integrable.

Both tested schemes retain dual-rail neurons. The comparison is one versus two
devices per physical dual-rail edge, corresponding to four versus eight
devices per original teacher synapse.

The simulated signed interaction is in
[`SignedDenseResistive`](../model/resistive/interaction.py). The active port
equations, power implications, and residual criterion are detailed in
[Differential pairs using signed amplifier ports](differential_pair_amplifier_implementation.md).
