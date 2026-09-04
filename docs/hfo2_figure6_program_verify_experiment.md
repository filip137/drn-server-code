# Figure-6 HfO2 program-and-verify follow-up

This exploratory, non-canonical follow-up adds stochastic program-and-verify
to the synthetic Figure-6-informed HfO2 DRN model. It freezes the direct
teacher target and the selected endpoint-population-HWA targets from the
September 3 pilot rather than selecting new logical targets after observing
P&V outcomes.

The source pilot is
`simulation_results/hfo2_figure6_drn/20260903T162705.930811Z-8c10965a-7f934fe4`.
Its scientific-summary SHA-256 is
`1e4d8b5c0dd0357ca7613ffdf6097a88d5c73559df43e17bc160489960e192c8`.
The selected independent/rank-matched HWA checkpoints have SHA-256 values
`76bed2f41c71f216971e4c4ed616bfe93656ef5f36ae8215f08652e4525c8551`
and
`580a797af53566e37605341da88622b409bb544be87e9637339041c49e9915c3`.

## Frozen four-arm comparison

For each independent and rank-matched endpoint assignment, the comparison is:

1. direct logical target, then P&V;
2. endpoint-population-HWA logical target, then P&V;
3. open-loop stochastic-pulse recovery from arm 1's exact persistent P&V
   endpoint; and
4. open-loop stochastic-pulse recovery from arm 2's exact persistent P&V
   endpoint.

“HWA + P&V” means P&V deployment of the HWA-selected logical target. The
source HWA trained with Figure-6 endpoint-population variation and exact
endpoint-affine gradients. It did not include pulse-resolved P&V residuals in
its training modifier. P&V-aware HWA would be a separate intervention.

## Program-and-verify controller

Every cell starts persistently at normalized progress `p=0`, so its full
conductance is its own synthetic `full_tile_RESET` endpoint. Exact `p=0`
targets are therefore already realized and are preaccepted without any
target-programming pulse. This is essential for the intended sequence: P&V
programs active rails from the known full-RESET initialization rather than
moving inactive rails away from RESET in response to an out-of-bound noisy
observation. The controller receives one noisy apparent RESET observation for
the remaining targets and then only:

- the continuous target progress;
- apparent normalized verify values; and
- its own pulse history.

It applies one stochastic HfO2 SET or RESET pulse, verifies again, and stops
when the apparent residual lies in

```text
abs(p_apparent - p_target) <= 0.11555.
```

The half-window is half the nominal central HfO2 progress increment
`0.2311`. The target-programming cap is 128 pulses per cell. The frozen pulse
parameter/noise seeds are 94001/94002 and are shared by direct and HWA targets
as common-random-number controls. The simulator saves apparent acceptance,
persistent-in-window success, budget exhaustion, reversals, SET/RESET counts,
and persistent/apparent residuals separately.

The persistent endpoint is the physical continuation state. The last
apparent verify observation is never substituted for it. Because apparent
write noise can leave normalized `[0,1]`, the primary network diagnostic
projects the held apparent progress to `[0,1]` before applying the positive
conductance map. The literal unprojected apparent state and a secondary
persistent-state accuracy are also saved. This is an explicit passive-DRN
diagnostic, not a claim that AIHWKit clips its apparent weight. There is no
independent inference-read-noise model, so fresh-read correctness is recorded
as unavailable rather than inferred from write noise.

An initial version-1 diagnostic used a symmetric apparent window even for
exact RESET targets. It pulsed 31,012 of the 79,400 already-correct inactive
rails and moved 21,248 away from RESET in the direct arm. That boundary rule
does not implement the intended initialization sequence and is superseded by
the preaccepted-RESET version-2 contract. Its artifacts are retained at
`simulation_results/hfo2_figure6_program_verify/20260904T091537.884394Z-614ca6d0-959b13ee`
rather than relabelled.

## Open-loop recovery

Each recovery arm bitwise clones its named persistent P&V endpoint and
regenerates the same device-to-device HfO2 pulse parameters. P&V pulse counts
do not consume the separate 64-pulse recovery cap. Recovery uses common
pulse-noise and Bernoulli-selection seeds 95002 and 95003.

Digital BPTT and Adam select probabilistic pulse commands. The weight state
changes only through stochastic HfO2 SET/RESET pulses, but there are no
verify reads or target-feedback controller during recovery. The accurate
claim is **hardware-in-loop open-loop stochastic-pulse recovery**, not
autonomous on-chip learning or program-and-verify recovery.

The `pilot` profile retains the prior 8,192-example training subset,
2,048-example validation cohort, full 10,000-example test set, three recovery
epochs, common solver/gain, and identical minibatch order in every recovery
arm. Validation on the held apparent state selects the recovery epoch; test
labels are not used for selection. The target assignment was already visible
during prior method development, so the outcome remains development evidence.

The implementation is
[`hfo2_figure6_program_verify_experiment.py`](../experiments/mnist_relu_drn/hfo2_figure6_program_verify_experiment.py).
Run its smoke or pilot profile from the repository root with:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_relu_drn.hfo2_figure6_program_verify_experiment \
  --profile smoke
```

## Corrected CUDA pilot result

The version-2 `pilot` completed locally on the RTX 3090 at
`simulation_results/hfo2_figure6_program_verify/20260904T092319.858588Z-5e63cfc5-40f94630`.
It evaluated the full 10,000-example MNIST test set. The scientific-summary
SHA-256 is
`340a9d2587288373e11cadd6ee6e2a2e2574e5a76944743e2e7d029d3469e719`.

| Endpoint pairing | Logical target | Exact target | P&V apparent | P&V persistent | Selected recovery apparent | Selected recovery persistent | Recovery epoch / pulses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent | Direct | 95.85% | 75.88% | 72.03% | 74.84% | 88.28% | 2 / 16,996 |
| Independent | Endpoint-noise HWA | 91.70% | 70.81% | 79.60% | 65.47% | 87.67% | 2 / 16,695 |
| Rank matched | Direct | 94.83% | 66.60% | 69.11% | 74.51% | 88.64% | 1 / 8,978 |
| Rank matched | Endpoint-noise HWA | 93.18% | 80.85% | 84.35% | 82.78% | 90.44% | 1 / 8,133 |

“Apparent” is the held noisy observation projected to `[0,1]`; “persistent”
is the underlying state used to continue pulse updates. Validation selected
the recovery epoch using apparent-state accuracy, so persistent improvements
are paired diagnostics rather than a persistently selected upper envelope.

P&V used 156,539 pulses for either direct target and 177,498/179,885 for the
independent/rank-matched HWA targets. No exact RESET target received a
target-programming pulse. Among non-RESET targets, apparent acceptance was
99.91% for direct and 99.85/99.83% for HWA, while persistent-in-window rates
were only 73.45% and 70.44/70.47%. There were 71/121/71/132 cap-exhausted
cells out of 158,800. Thus the high apparent acceptance does not imply that
the persistent cell state reached the requested tolerance.

Across the two pairing regimes, mean persistent accuracy was 70.57% for
direct P&V and 81.98% for HWA-target P&V. The selected open-loop recovery
raised those means to 88.46% and 89.06%, respectively. HWA therefore improved
the mean programmed persistent state in this one fixed target assignment, but
after recovery its mean advantage was only 0.60 percentage point. The
pairing-specific spread is large, and the target identities were not sealed,
so this is mechanism evidence rather than a robust HWA or deployment claim.

About 51,000 held apparent values per arm lay outside the modeled endpoint
interval, overwhelmingly below RESET, and were projected for the passive-DRN
apparent diagnostic. Consequently the apparent numbers are highly sensitive
to this explicitly declared projection and must not be interpreted as a
literal physical-conductance measurement. The persistent comparison is the
cleaner readout of whether P&V plus open-loop pulses repaired the stored state.

All five declared artifact hashes were verified. All four checkpoints reload
with `torch.load(..., weights_only=True)`, reproduce P&V pulse accounting,
retain all 79,400 exact-RESET cells unchanged, and contain a bitwise exact
persistent P&V-to-recovery handoff with recovery pulse counts reset to zero.
