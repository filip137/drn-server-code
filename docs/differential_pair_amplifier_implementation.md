# Differential pairs using signed amplifier ports

This note describes an ideal **active two-port** realization of a differential
ReRAM edge. It is an equation-level circuit contract, not yet a transistor-level
implementation. In particular, the returned-current path below must be built
explicitly; it is not supplied by an ordinary voltage buffer.

For an interior edge from nonlinear node `x` to next-layer node `y`, use two
matched voltage-output ports:

```text
                         G+
x -> nonlinearity -> +A x ----\
                               +-- y
                   -> -A x ----/
                         G-

        sense both output currents and return their signed sum to x
```

Let `A` and `B` be arbitrary finite positive magnitudes. The positive port has
signed gains `(A_+, B_+) = (+A, +B)` and the negative port has
`(A_-, B_-) = (-A, -B)`. For `s` equal to `+1` or `-1`, define

```text
u_s     = s A x
j_s     = G_s (y - u_s)       current leaving y toward output port s
i_x,s   = -s B j_s            current returned at x
```

The last equation is the required current-sensing and signed-feedback law. It
can be represented by a current-controlled current source in a behavioral
circuit. A high-input-impedance VCVS by itself would give no returned current
and would therefore not implement this edge. One amplifier pair may be shared
by all outgoing synapses of a node only if the two aggregate output currents
are sensed separately and the prescribed signed aggregate is returned to that
same pre-amplifier nonlinear node.

## Raw KCL and loading

Write

```text
S = G+ + G-
D = G+ - G-
```

For the IBM OM differential-device notation, identify `G+ = a` and `G- = r`.
The represented coupling is therefore `D = a-r`, while the conductance in the
voltage denominator and both endpoint self-loading terms is `S = a+r`. The
signed effective weight `D` must not replace `S` in those denominator terms.

With `i_x` and `i_y` both defined as currents leaving their respective nodes
into the edge, the two endpoint currents are

```text
i_x = A B S x - B D y
i_y =       S y - A D x.
```

Thus the raw two-node admittance contribution is

```text
        [ A B S   -B D ]
Y_edge =[               ].
        [  -A D      S  ]
```

This matrix is generally nonreciprocal: its two transfer entries differ when
`A != B`. Calling `A` and `B` inverse or matched gains must not be confused
with electrical reciprocity, passivity, or power conservation. The output
stages and signed current-return sources are active and require power rails.

For a nominal zero weight, `G+ = G- = G_floor`, so `D = 0` and

```text
i_x = 2 A B G_floor x
i_y = 2       G_floor y.
```

The common floor therefore cancels from inter-node transfer for any positive
`A` and `B`, but remains as loading at both endpoints. The special choice
`A B = 1` equalizes the source-side self-loading coefficient with the
destination-side coefficient; it is **not** required for the existence of an
energy representation. With `A = 4` and `B = 0.25`, the older formulas
`i_x = S x - 0.25 D y` and `i_y = S y - 4 D x` are recovered.

The physical device dissipation is

```text
P_devices = G+ (A x - y)^2 + G- (A x + y)^2.
```

At equal floor it is `2 G_floor (A^2 x^2 + y^2)`. Consequently, cancellation
of the represented zero weight does not cancel device current, amplifier
current, static conductance, or power. In a high-fan-out row, the common floor
can still impose a large output-current and supply-power burden.

## Weighted energy for a layered network

Although the raw KCL matrix is nonreciprocal, it is diagonally integrable when
`A` and `B` are finite and positive. Number the first dynamic layer after the
clamped input as layer `1` and set

```text
m_1 = 1
m_(l+1) = m_l B / A.
```

For an interior edge from `x` in dynamic layer `l` to `y` in layer `l+1`, let
`m_post = m_(l+1)` and define

```text
E_edge = 0.5 m_post [G+ (A x - y)^2 + G- (A x + y)^2].
```

Using `m_l = m_post A / B`, direct differentiation gives

```text
dE_edge/dx = m_l    i_x
dE_edge/dy = m_post i_y.
```

Hence the physical KCL field is not itself the gradient in raw voltage
coordinates, but multiplication by the positive layer metric converts it into
one. Equivalently, the global weighted Jacobian is symmetric even though the
raw admittance matrix is not. `A B = 1` plays no role in this integrability
condition.

The clamped-input boundary is unity gain. If fixed input `u` connects to the
first dynamic voltage `y_1`, use

```text
E_input = 0.5 m_1 [G+ (u - y_1)^2 + G- (u + y_1)^2],
m_1 = 1.
```

The input is not optimized, so no input-side returned-current equation is
needed. Its derivative with respect to `y_1` is exactly the first-layer KCL
contribution. Local diode/current-law primitives, biases, and any energy-based
nudging terms must likewise be scaled by the metric of their dynamic layer if
the total scalar energy is to generate the same physical equilibrium.

The current simulator deliberately supports a narrower, fail-closed envelope
than the equation-level construction: every dense edge must use the
differential pair; convolution, low-rank adapters, biases, and historical
process-global amplifier indexing are rejected; and the local nonlinearity
must be `linear` or `perfect_diode`. Amplifier magnitudes and layer metrics
must remain finite and positive in the selected tensor dtype, and both branch
conductances must use non-negative physical bounds. These are implementation
limits, not an additional mathematical requirement on `A B`.
An unscaled abstract cost remains a valid scalar nudging energy, but corresponds
to physical injected current `grad C / m_l`; hardware intended to inject the
same specified current in raw KCL coordinates must multiply its energy term by
`m_l`.

## Residual and KKT criterion

Let `r_l(v)` be the unscaled physical KCL residual at dynamic layer `l`, after
including every incident edge and local nonlinear current. The corresponding
energy gradient is

```text
grad_l E(v) = m_l r_l(v).
```

For an unconstrained node, equilibrium is `r_l = 0`, equivalently
`grad_l E = 0`. For an ideal-diode or otherwise constrained voltage
`v_l in C_l`, the correct condition is

```text
0 in r_l(v) + N_C_l(v_l),
```

where `N_C_l` is the normal cone of the feasible voltage set. Because
`m_l > 0` and a normal cone is closed under positive scaling, this condition is
equivalent to the weighted-energy KKT condition
`0 in grad_l E(v) + N_C_l(v_l)`.

A circuit-versus-solver comparison should therefore report the raw residual
`r_l = grad_l E / m_l`, or an equivalent projected KKT residual, rather than
compare unnormalized energy-gradient magnitudes across layers. The metric may
change sharply with depth, so a small weighted gradient alone can hide a large
physical KCL error.

## Physical qualifications

Exact cancellation assumes matched positive/negative voltage gains, matched
signed current-return gains, linear bilateral device conductance, zero output
impedance, adequate voltage and current compliance, and negligible line
resistance. Gain mismatch converts the common conductance into residual
coupling and also perturbs endpoint loading. Device mismatch, programming
error, relaxation, and read noise leave a residual `G+ - G-`. Nonlinear or
polarity-asymmetric ReRAM read current prevents exact large-signal cancellation
unless it is included explicitly in the branch equations.

Finite amplifier bandwidth and the active current-feedback loop also make
closed-loop stability, noise, and settling circuit questions. These require a
controlled-source SPICE realization followed by a transistor-level design and
cannot be inferred from `A B = 1`.

This topology realizes the signed branches described in
[Differential memristor scheme](differential_scheme.md) after the layer metric
and boundary convention above are applied. Dual-rail neuron coding remains a
separate architectural choice; when retained, one signed amplifier pair is
attached to each physical excitatory or inhibitory dynamic node.
