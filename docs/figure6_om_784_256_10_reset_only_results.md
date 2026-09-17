# Figure-6/OM RESET-only hidden-SET exploratory result

Status: complete exploratory screen (`exploratory_noncanonical`), 2026-09-05.

## Question and information boundary

This screen asks whether the `784-256-10` DRN can be deployed when each
cell's initial Figure-6 RESET conductance is known but its final SET
conductance is hidden from the mapper and program-and-verify controller.

For each logical four-cell quad, the mapper used

\[
B_Q=\max_i R_i, \qquad A_Q(m)=B_Q+m(2.0-B_Q),
\]

and requested one common baseline and one common active conductance. Positive
weights used `[A,B,B,A]`; negative weights used `[B,A,A,B]`. Consequently, the
requested physical targets satisfy `G++ = G--` and `G+- = G-+` exactly. The
fixed nominal SET value `2.0` is a population-level assumption, not knowledge
of any cell's true SET.

The hidden true SET was available only to the simulated physical plant: it
could clip an unreachable target but could not resize the request, enter the
HWA gradient, or enter the P&V controller. P&V observed absolute apparent
conductance, not true-SET-normalized progress. All primary forwards used the
current held apparent state. Persistent state remained write authority and
was evaluated only as a secondary diagnostic on the same examples.

This run retained the previous screen's teacher, topology, data split, solver,
logit gain, endpoint populations and seeds, OM identities, write seeds, and
fixed HWA learning rates. It used three held-out arrays and three independent
programming repetitions per array. HWA ran for ten epochs and selected epoch
7 using development endpoints only. No Adam recovery was run in this focused
screen.

## Primary apparent-state results

The values below are hierarchical means across the three held-out arrays;
brackets give the range of the three array means. KL is
`KL(teacher || student)`.

| Arm | Accuracy | KL divergence |
|---|---:|---:|
| Direct P&V | 97.9122% [97.8800, 97.9367] | 0.039864 [0.039750, 0.039926] |
| Direct P&V + RESET-stuck corruption | 97.0167% [96.8867, 97.2600] | 0.098955 [0.092249, 0.102660] |
| HWA + P&V | 97.8456% [97.8033, 97.8833] | 0.012575 [0.012094, 0.013177] |
| HWA + P&V + RESET-stuck corruption | 97.1389% [97.0267, 97.2167] | 0.034175 [0.033648, 0.035086] |

On clean arrays, HWA changed top-1 accuracy by -0.0667 percentage points but
reduced KL by 0.027289. Under corruption, HWA was +0.1222 percentage points
more accurate than direct deployment and reduced KL by 0.064780. Corruption
cost 0.8956 percentage points for direct deployment and 0.7067 percentage
points after HWA.

Before P&V, the ideal plant-limited RESET-only mappings also remained strong:
direct deployment averaged 97.9367% accuracy and 0.024555 KL; HWA averaged
97.8200% and 0.011517 KL. These are deterministic hidden-SET-clipped views,
not programmed-device results.

## Why unknown SET did not collapse accuracy

Unknown SET only matters when a common requested target exceeds a cell's
physical ceiling. That occurred rarely:

| Family | Requested targets above hidden SET | Approximate cells |
|---|---:|---:|
| Direct | 0.0395% | 321 |
| HWA | 0.3572% | 2,904 |

The requested pair-equality residual was exactly zero. Hidden SET clipping
introduced only modest ideal physical mismatch: across the two layers, RMS
`G++-G--` was 0.0040--0.0061 for direct and 0.0074--0.0174 for HWA; RMS
`G+--G-+` was 0.0072--0.0127 for direct and 0.0156--0.0300 for HWA.

After stochastic P&V, RMS pair mismatch was about 0.057--0.060 for direct and
0.059--0.066 for HWA. A read-only audit of the earlier exact-endpoint run found
that its equal-normalized-progress mapping already requested RMS physical pair
mismatches of about 0.112--0.149 before programming, increasing to
0.138--0.170 in the apparent programmed state. Thus the shared physical
conductance target removed a much larger error source than the rare hidden-SET
saturations introduced.

This is the central explanation: the old mapping knew every SET endpoint but
matched normalized progress, so equal logical device codes did not imply equal
physical conductances. The new mapping knows less about individual cells but
enforces the cancellation identities in physical conductance wherever the
targets are reachable.

## Comparison with the previous exact-endpoint screen

| Arm | Previous accuracy | RESET-only accuracy | Change | Previous KL | RESET-only KL | Change |
|---|---:|---:|---:|---:|---:|---:|
| Direct P&V | 97.5133% | 97.9122% | +0.3989 pp | 0.032821 | 0.039864 | +0.007043 |
| Direct + corruption | 96.2789% | 97.0167% | +0.7378 pp | 0.088237 | 0.098955 | +0.010718 |
| HWA + P&V | 97.7156% | 97.8456% | +0.1300 pp | 0.030308 | 0.012575 | -0.017733 |
| HWA + corruption | 97.0456% | 97.1389% | +0.0933 pp | 0.058703 | 0.034175 | -0.024527 |

These deltas must **not** be interpreted as the isolated value of SET
knowledge. The new screen simultaneously hides per-cell SET and replaces the
old per-cell normalized mapping with a shared physical-target mapping. It is a
useful deployment result but a composite intervention.

The matched causal follow-up should hold the shared-baseline/shared-physical-
target rule fixed and compare two target ceilings:

1. an oracle ceiling derived from the active cells' true SET values; and
2. the present nominal `2.0` ceiling with true SET hidden.

That comparison would isolate whether per-cell SET knowledge adds value once
the physical cancellation constraint is already enforced. A two-cell-shared
baseline arm should then be compared separately; baseline sharing must not be
confounded with the SET-information boundary.

## P&V and state diagnostics

Controller completion was 99.9703% for direct targets and 99.8152% for HWA
targets. Among mathematically unreachable requests, apparent noise caused
57.9% (direct) and 53.9% (HWA) to be accepted by the controller; the remainder
exhausted the 128-pulse budget. This is apparent acceptance, not persistent
reachability.

The secondary persistent-state results were lower than the primary
apparent-state results:

| Arm | Persistent accuracy | Persistent KL |
|---|---:|---:|
| Direct P&V | 97.4344% | 0.084443 |
| Direct + corruption | 95.5100% | 0.173950 |
| HWA + P&V | 97.5400% | 0.023421 |
| HWA + corruption | 96.2044% | 0.063905 |

That gap is expected under the declared held-write-noise model and reinforces
that the apparent state must remain the primary forward state. It does not
license using persistent-state accuracy for optimization or selection.

## Provenance and verification

Completed run:
`simulation_results/exploratory_noncanonical_figure6_om_784_256_10_reset_only/20260905T162731.179447Z-36e3e5a5-34d692da`

- Source commit: `2b283365063931e9fdd41fbd30d2672e0e600f04`
- Dirty-worktree hash recorded by run: `22fb524c702fa3904e9070378d9ead24c5a641294e61f786aed12124974da669`
- Config SHA-256: `f45efc8fa126b8fc2e6e54148b8ee6593881dd4e6573db72551cbced85c762ec`
- Scientific summary SHA-256: `aba0f386fd4d79f147a95b42a29f018e6467ebe44d507a30dbfbb255526bb201`
- Raw records SHA-256: `22c90be1f97a8fddb1835ef2c60d68482a697695477d809c13c3a6f2005d2476`
- Selected HWA checkpoint SHA-256: `c632accbdc3fd7b9d64c6b5e2bb0986c79edfae90cac65d8cf83ca77bf426bbc`

The real host-level CUDA allocation/kernel/synchronization canary passed on
the NVIDIA GeForce RTX 3090. Focused verification passed 14 tests covering
RESET-only target equality and SET independence, hidden-SET plant response,
SET-free gradient lifting, absolute-conductance P&V, the existing ladder
config, and pulse behavior. The complete run produced 6 ideal records and 36
P&V records with all declared arrays, write seeds, families, and corruption
arms present.
