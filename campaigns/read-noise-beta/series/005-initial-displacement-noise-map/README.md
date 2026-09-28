# Initialization: normalized displacement versus endpoint noise

**Completed September25:** the full screen and five-draw confirmation validate
selected worst-layer advantages for ours in Conv1/2; Conv3's tiny candidate is
mixed and not confirmed. No all-layer dominance or training benefit is established.
All57 screen/confirmation contexts validate, with no failures or exclusions;
approximately0.97GPUh total and both GPU lanes released. See the
[screen review](results/exp-010-initial-cosine-surface.md),
[final confirmation review](results/exp-011-confirm-candidate-region.md) and
[report/plots](../../../../results/conv123-initial-displacement-noise-map-20260925-v1/report.md).
No further simulation or trained-checkpoint replay is launched.

Question: is there a region where ours or baseline estimates its clean BPTT gradient
more accurately than legacy, despite legacy's observed noisy-training advantage?
[H008](../../hypotheses/H-008-initial-alignment-region.md) is the prospective claim;
[X007](../../explorations/X-007-initial-displacement-noise-map.md) explains the controls.

This series starts only at the shared seed-0 initializations of Conv1/2/3. Its
instrument is read-only layerwise EqProp/BPTT cosine at fixed T/K4/6/8, not training.
The x-coordinate is matched pooled output D/F; the y-coordinate is absolute Gaussian
endpoint read-noise sigma. Do not pool trained checkpoints, absolute-D matching,
relative-noise models or different reference horizons into these surfaces.

| Stage | Experiment | Bounded work |
|---|---|---|
| Reuse and match | [exp009](experiments/exp-009-reuse-and-match.md) | Audit existing betas; reuse27 target matches, verify only18 missing predictions at0.2/10. |
| Coarse surface | [exp010](experiments/exp-010-initial-cosine-surface.md) | Five displacement targets × five nonzero noises plus clean; all depths/schemes/layers; three noise draws. |
| Conditional confirmation | [exp011](experiments/exp-011-confirm-candidate-region.md) | At most one adjacent pair per depth; five fresh noise draws; no new training. |

Targets r:0.2,0.8,2,6,10. Noises sigma:0,1e-4,5e-4,1e-3,2e-3,5e-3.
Reuse exact matched evidence when source, initializer, cohort, T/K, loss, gradient
coordinates and noise draws agree; otherwise use historical values only as beta
proposals. Cache clean state solves and BPTT before adding endpoint noise.

Deliverables: measured beta/D/F lookup with provenance; per-layer absolute-cosine
and difference heatmaps; clean-versus-noisy decomposition, worst-layer summary,
validity/coverage tables and a conclusion scoped to the sampled domain.

## Execution assignment — September 25

Filip explicitly requested launch on **local GPU, akibscomputer, and nom-cool-1
if available**. This supersedes the original planning-only status and single-host
preference for the whole series. Keep each architecture's entire scheme/target/noise
surface on one host/runtime; do not split matched schemes across hosts. Proposed
placement is Conv1 Akib RTX3080, Conv2 nom-cool-1 RTX3090, Conv3 local RTX3090,
subject to live admission and measured memory. If nom-cool-1 is unavailable, queue
one complete architecture on another requested host without disturbing unrelated jobs.
Host-dependent timings and RNG/runtime identities remain explicit across depths.

Authorized budget6.5 GPUh total:0.5 missing-target checks,4 screen including smokes,
2 conditional confirmation. Divide the4h screen pool between architectures from
measured smoke timings, not4h per GPU. Immediate daytime execution is authorized;
wall deadline **September25 20:00 Europe/Paris**, with no overnight extension.
Expected compute is roughly1–3 elapsed hours after implementation/preparation with
parallel hosts, subject to the real smoke benchmark; this is not a measured ETA.
No additional T/K sweeps, optimizer steps, trained states or official-test reads.

Result root `results/conv123-initial-displacement-noise-map-20260925-v1/` was
persistently indexed before creation. Preparation/numerics: relative_output_prepare;
transport/admission/recovery: relative_output_capacity; monitoring: clean_beta_monitor;
coordinator/review: root. Fifi/Loulou stay released. The Codifier must record exact
commands, source/config hashes and per-host handles; Monitor acknowledges executable
handoff before ownership transfers. Scope includes the bounded conditional
confirmation if the predeclared candidate rule is satisfied; no new training.

## Follow-up observation

Filip finds legacy's retained higher later-layer cosine surprising given expected
backward attenuation of the nudging signal. [X008](../../explorations/X-008-legacy-alignment-despite-attenuation.md)
records his observation, the measured layerwise exceptions, and the distinction
between matched D/F and unequal absolute signal under absolute endpoint noise.
This is an open mechanism question; no new runs are assigned.
