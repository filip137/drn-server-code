---
experiment: "exp-007"
evidence: "validated-local"
summary: "33 completed cases plus three user cancellations; ours-minus-legacy means are -0.065pp for Conv1 and -4.460pp for Conv3, while Conv2 lacks its fourth pair."
verdicts: {"H-006": "inconclusive"}
---
# Matched initial relative output displacement: exp-007

All 36 original cases are reconciled: **33 completed and three explicitly
user-cancelled Fifi cases**. All 35 attempted canonical bundles validate locally;
33 are successful, one is a partial user cancellation, and one operational
Conv3 attempt is preserved and replaced under the same scientific contract.
The two other cancelled arms never started. All 33 included outcomes pass the
scientific config, source, initialization, epochs, dataset/order, noise, bias,
finite-checkpoint and placement checks. No official-test read occurred.

Seed 0, ordinary MNIST 55k/5k, ten epochs, centered frozen-current EqProp,
float64, physical paired 20-node output, frozen zero biases, source Adam rates,
T=K of 4/6/8, and additive endpoint sigma5e-4 are held fixed. All 36 initial
D/F matches passed the 1% tolerance; beta remained fixed during training.

| Architecture | Ours − legacy at targets 0.8 / 1.2 / 1.6 / 2.0 (pp) | Four-target mean (pp) | Scoped H-006 verdict |
|---|---|---:|---|
| Conv1 | −0.06 / −0.10 / −0.02 / −0.08 | −0.065 | contradicts |
| Conv2 | −0.80 / −0.50 / cancelled / −0.36 | undefined | inconclusive |
| Conv3 | +4.44 / −6.68 / −13.06 / −2.54 | −4.460 | contradicts |

Ours wins at zero of four Conv1 targets and one of four Conv3 targets. All
three observed Conv2 differences favor legacy, but its required fourth pair
is missing, so no partial mean is substituted. The broad complete-grid
verdict remains **inconclusive** under the experiment's explicit cancellation
clause; the complete Conv1 and Conv3 scopes contradict the proposed average
advantage, and these data do not support an all-depth advantage for ours.

The [full report](../../../../../results/eqprop-conv123-relative-output-20260924-v1/analysis/report.md)
contains every baseline/ours/legacy endpoint, all 12 intended contrasts,
losses, trajectories, identities and exclusions. No best target or best epoch
was selected. Conv3 legacy at target 0.8 falls from 83.78% best to 78.06%
final, failing the separate five-point stability screen; this finite negative
outcome remains included. Conv3 ours at 1.6 finishes at 75.80%.

Fifi baseline target 1.6 stopped around epoch 6, batch 3000; ours/legacy never
started, and no restart is authorized. Conv3 baseline 1.6's first attempt,
`jz-6`, suffered severe node paging; it is explicitly excluded operationally
and preserved beside validated `jz-6-recovery1`. The first two epoch metric
dictionaries match exactly. Recovery canary and all smokes are excluded from
accuracy evidence. Source archive SHA256 is
`19f297708ca43a699398e17791c90ddd12981beb3bfffa1af5193cae0ec58d1c`.
Authoritative bundles are under the study's `production/local` and
`collected/{akib,riri,loulou,trex,fifi,jean-zay}/production` roots.

Production/recovery consumed approximately 26.662 GPU-hours of allocation or
physical host occupancy; known calibration and local smokes bring observed use
to roughly 26.8 GPU-hours, below the 60-hour cap. Concurrent Conv2 processes
are counted once per physical host interval, with host clock offsets retained.

This is descriptive single-seed validation evidence, not a significance claim
or paper test result. Initial D/F does not equalize absolute displacement,
hidden-layer signals or SNR under absolute additive noise. Fixed beta allows
D/F to drift; inherited scheme-specific rates and between-target host changes
remain limitations. The result tests this normalization policy, not isolated
amplification causality.

Review is finished; the experiment state is **partial** because three requested
trainings were cancelled. The already authorized
[exp-008 extension](../experiments/exp-008-relative-output-noise-extension.md)
continues under its own declared target/noise rule. Its selection after partial
exp007 evidence does not change this original decision rule or authorize
restarting cancelled cases.
