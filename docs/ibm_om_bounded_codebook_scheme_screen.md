# IBM OM deterministic bounded-codebook deployment screen

- Status: predeclared and implemented; full execution pending
- Screen ID: `mnist-ibm-om-bounded-codebook-scheme-screen-20260827-v1`
- Evidence class: hardware-derived fitted AIHWKit model
- Source: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`

## Question and hypothesis

The first identity-aware deployment test asks whether each of the four
reference-policy-by-topology schemes can retain the frozen teacher's accuracy
when the deployment uses literal OM identities, their ranges, and their
deterministic pulse resolution, but no stochastic write error.

The hypothesis is that every scheme retains at least 90% test accuracy on the
mean of three frozen held-out assignments. This is a gate, not an assumption:
any failing scheme remains blocked from stochastic program-and-verify and
HWA.

## Matched arms

The intervention crosses two independent axes:

| Arm | Physical representation |
| --- | --- |
| Four devices, no fitted `r` | one conductance on each dual-rail edge, based at that identity's pulse-zero lower code |
| Four devices, fixed `r` | positive logical weights use `[[a,r],[r,a]]`; negative weights swap the roles |
| Eight devices, no fitted `r` | one active and one lower-code branch per dual-rail edge |
| Eight devices, fixed `r` | one active and one fixed-reference branch per dual-rail edge |

In every arm, physical conductances remain separate. Four-device edge loading
is the programmed edge conductance. Eight-device transfer is `G_a-G_r` and
loading is `G_a+G_r`. The signed difference is never used in a voltage
denominator.

The four- and eight-device populations are deterministic topology-specific
draws under the same assignment seed, but their hashes differ because their
binding names and physical cell counts differ. Within each topology, the
with- and without-reference arms reuse exactly the same identities. Any
cross-topology difference is therefore interpreted over the three assignment
replicates rather than as a per-cell paired effect.

## Ideal bounded-codebook contract

The logical source and DRN numerical computation remain FP32. For every
assigned cell:

1. retain its sampled `a_min`, `a_max`, `r`, and SET response parameter;
2. start exactly at `a_min`;
3. enumerate pulse indices 0 through 128 with the AIHWKit-transcribed SET
   equation, setting the cycle-to-cycle random draw to zero;
4. set apparent write noise to zero;
5. map every state through the global model coordinate
   `x=clip((a+1)/2,0,1)`; and
6. choose the code with the smallest absolute target error, breaking exact
   ties toward the lower pulse index.

The repaired corruption policy is frozen to isolate range and resolution.
Published corrupt cells are replaced only by the repository's deterministic
donor rule and remain counted. A fitted reference outside its assigned cell's
range is not treated as a negative conductance: it is projected to the nearest
reachable deterministic code and the event and error are reported.

This codebook is device-specific. The nominal `dw_min` is not reinterpreted as
a universal level count. No analyst-selected 9-level logical quantizer, pulse
controller, endpoint sampler, HWA modifier, optimizer update, or recovery rule
is active.

## Frozen comparison

All arms use the same bias-free ReLU teacher, normalized logical matrices,
MNIST cohorts, perfect-diode DRN, solver, and fraction grid. Assignment
`86001` selects a layer-fraction pair by bounded-codebook calibration accuracy,
then calibrated KL and lexicographic scales. The selected fractions and gains
are frozen before assignments `87001`, `87002`, and `87003` are sampled and
evaluated.

For each selected map, an exact continuous target inside the same codebook
baseline-to-maximum envelope is evaluated as a diagnostic upper control. It
does not choose a different range or scale. Its gap to the codebook result
isolates nearest-level projection loss.

## Completion and analysis

Completion requires all four arms on the development assignment and all three
held-out assignments, exact population receipts, complete target hashes, and
deterministic replay. Report accuracy, teacher agreement, KL, continuous-to-
codebook prediction flips, reference projections, pulse indices, effective
level counts, target error, signed contrast, sign flips, denominator loading,
voltage statistics, and total-conductance power proxies.

The headline is the mean and full range of held-out bounded-codebook accuracy
for each arm. Interpret four-versus-eight and with-versus-without-`r`
differences only within this matched screen.

## Claim boundary

The OM preset is a hardware-derived normalized fit, not raw device-trace
replay. It has no unique microSiemens origin or scale. Results therefore show
behavior under the declared normalized OM model and deterministic oracle
codebook; they do not demonstrate fabricated-device accuracy, realistic
programming success, absolute voltage/current/power, or a need for on-chip
training.
