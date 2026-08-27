# IBM OM deterministic bounded-codebook deployment screen

- Status: corrected shared-calibration v2 exploratory execution complete and
  deterministically replayed. The completed v1 exploratory run is superseded
  because it fitted calibration separately in each arm.
- Screen ID: `mnist-ibm-om-bounded-codebook-scheme-screen-20260827-v2`
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
MNIST cohorts, perfect-diode DRN, solver, and one calibration derived before
the scheme intervention. The immutable initial-RESET receipt
`data/ibm_om_cell_aware_full_span_v1.receipt.json` (SHA-256
`d58b0d709312061dc2425d143744f45fdf9de9d5e583a619e7a91325c53059b8`)
fixes layer fractions `[1.0, 1.0]` and positive output gain
`4.46683592150963`. Its teacher hash must match the screen teacher and its
`optimizer_updates` field must remain zero.

Those fractions and that gain are applied unchanged to all four schemes and
to both the bounded-codebook and continuous-envelope states. No variant is
allowed to select a scale or fit a gain. Assignment `86001` now supplies only
matched development diagnostics; assignments `87001`, `87002`, and `87003`
are held-out evaluations. The shared positive gain cannot change top-1
predictions, but freezing it prevents scheme-specific KL rescaling from being
mistaken for deployment performance.

For each selected map, an exact continuous target inside the same codebook
baseline-to-maximum envelope is evaluated as a diagnostic upper control. It
does not choose a different range or scale. Its gap to the codebook result
isolates nearest-level projection loss.

## Completion and analysis

Completion requires all four arms on the development assignment and all three
held-out assignments, validation of the exact initial-RESET receipt, exact
population receipts, complete target hashes, and deterministic replay. Report
accuracy, teacher agreement, fixed-gain KL, continuous-to-codebook prediction
flips, reference projections, pulse indices, effective level counts, target
error, signed contrast, sign flips, denominator loading, voltage statistics,
and total-conductance power proxies.

The headline is the mean and full range of held-out bounded-codebook accuracy
for each arm. Interpret four-versus-eight and with-versus-without-`r`
differences only within this matched screen.

## Exploratory v2 result

All four arms used the same `[1.0, 1.0]` fractions and fixed gain
`4.46683592150963`. The three-assignment held-out results are:

| Scheme | Bounded-codebook mean (range) | Continuous-envelope mean | 90% gate |
| --- | ---: | ---: | --- |
| Four devices, no fitted `r` | 35.98% (27.36–43.72%) | 41.02% | fail |
| Four devices, fixed `r` | 57.85% (53.06–61.30%) | 64.35% | fail |
| Eight devices, no fitted `r` | 29.95% (23.44–35.17%) | 31.82% | fail |
| Eight devices, fixed `r` | 36.51% (31.81–40.92%) | 40.73% | fail |

At the deterministic codebook state, adding the fitted reference improves the
four-device mean by `21.87` percentage points and the eight-device mean by
`6.56` points. Moving from four to eight devices reduces the mean by `6.03`
points without `r` and by `21.34` points with `r`. The continuous controls
also remain far below the gate, so nearest-code projection is not the sole
failure mechanism.

The actual conductance-sum loading moves the network to very different voltage
regimes. Mean held-out hidden/output voltage RMS is `2.959/1.426` for four
devices without `r`, `0.313/0.054` for four devices with `r`, `1.947/0.668`
for eight devices without `r`, and `0.172/0.006` for eight devices with `r`.
This supports treating loading and the resulting operating point as part of
the scheme intervention; it does not show that symmetry subtraction alone is
the cause of the accuracy differences.

The corrected calibration does not change top-1 predictions: every v1 and v2
held-out bounded-codebook and continuous prediction hash is identical. That
is expected because all v1-selected fractions were already `[1.0, 1.0]` and
changing a positive output gain cannot change `argmax`. The correction does
matter for score RMS and KL: v1 applied four scheme-specific bounded-codebook
gains (`0.5012`, `0.7943`, `89.1251`, and `141.2538`), whereas v2 applies the
one RESET gain to every result.

The complete summary is the ignored exploratory artifact
`results/mnist-ibm-om-bounded-codebook-scheme-screen-20260827-v2/analysis/summary.json`
with SHA-256
`3fb84d925408942c44a3acf9e1a911f2c2633efd510c905c5f05fd1904ab869b`.
It contains two development and six held-out topology populations, four
development diagnostics, twelve held-out arm records, and eight exact
population receipts. A second complete execution reproduced the summary hash.

## Claim boundary

The OM preset is a hardware-derived normalized fit, not raw device-trace
replay. It has no unique microSiemens origin or scale. Results therefore show
behavior under the declared normalized OM model and deterministic oracle
codebook; they do not demonstrate fabricated-device accuracy, realistic
programming success, absolute voltage/current/power, or a need for on-chip
training.
