---
id: "exp-009"
title: "Reuse initialization displacement calibrations and check missing endpoints"
status: "complete"
hypotheses: ["H-008"]
---
# exp-009 — Reuse measured beta values

## Assignment and dependencies

Planning and source audit requested September25; subsequently Filip explicitly
authorized launch on local/Akib/nom-cool-1. Follow the series005 contract and exp007 frozen numerical
runtime. Existing initializers, calibrations and cohort must be readable/hash-valid.

## Cases and held-fixed contract

Nine architecture/scheme contexts, r=0.2/0.8/2/6/10:45 intended beta matches.
Reuse27 measured matches at0.8/2/6 from exp007/008. Choose an existing measured beta
within1% target error directly. Otherwise propose B_new=B_ref*r_target/r_measured,
using the nearest existing matched point, and measure once. Expected new checks:
18 points at0.2/10. Allow at most one ratio correction per unmatched prediction;
retain all measurements, then leave unresolved cases invalid instead of sweeping.
Do not repeat the27 accepted matches solely to create new result bundles.

D/F uses pooled squared values/counts over the existing576 validation examples in
36 batches of16 and the physical20 outputs. Matching tolerance1% relative error;
free RMS must be nonzero/usable. Clean phases, no optimizer/no endpoint noise during
matching. Measure all layers' D and free RMS as diagnostics at the chosen beta.
Preserve float64, explicit perfect-diode params, input gains40/100/360, frozen-zero
bias, weights[0,100], squared loss, source initializers, T=K4/6/8. Freeze beta after
this stage for all noise levels; base beta=B/(voltage_amp/current_amp)^depth.

## Sources and implementation

- `results/eqprop-conv123-relative-output-20260924-v1/calibration_summary.json`
- `results/eqprop-conv123-relative-output-noise-extension-20260924-v1/target6/calibration/all/`
- `configs/conv/eqprop_conv123_relative_output_20260924/`
- `configs/conv/eqprop_conv123_relative_output_noise_extension_20260924/`
- `experiments/calibrate_initial_relative_output.py` (existing runner to reuse).
- Clean Conv3 response: `results/conv3-clean-eqprop-rms-beta-20260924-v1/analysis/`.

Resolve exact raw bundle/checkpoint/config/cohort hashes and write a readable beta
lookup before execution. A numerically close beta from a different checkpoint,
D definition, horizon or cohort is an estimate, never an accepted match. Source
code/metadata changes after freeze require an explicit new identity. Reuse lookup
may be a small CSV, not a new catalog or launch framework.

## Budget, stop and outputs

Proposed cap0.5 GPUh including same-runner smoke; no training, official-test read,
T/K sweep or broad beta search. Prefer local RTX3090 after fresh admission. Preserve
nonfinite points and stop source/invariant failures. Planned result root:
`results/conv123-initial-displacement-noise-map-20260925-v1/`, child `beta_lookup/`;
register the top-level directory before creating it. No directory created by this plan.
Outputs: provenance table, actual B/base beta/F/D/r, deviations and coverage45/45 or
explicit missing entries. Summarize in `../results/exp-009-reuse-and-match.md` only
after actual validation. A matching result alone does not decide H008.

## Execution and monitoring handoff

Launch authorized September25; preparation underway. Codifier records config/command/source, target, budget and
result path after source audit, then uses existing smoke/reporting conventions.

## Existing cosine coverage inspected during planning

`results/eqprop-conv3-init-beta-noise-20260921-v1/analysis/report.md` contains the
same saved Conv3 seed-0 initializer and576-example cohort at injected B0.01/0.1,
T=K8, sigma0/1e-5/3e-5/1e-4/3e-4/5e-4/1e-3, one matched read-noise draw.
Its six cases and6,048 layer comparisons are historical measured evidence, not
samples at common D/F across schemes. They do not cover2e-3/5e-3 or the proposed
independent noise seeds. Reuse only exact compatible components; do not relabel
those cosines as the new matched-grid observations. The measured10.56-minute
runtime for that smaller six-case replay supports a compact replay rather than
another multi-day training sweep, but is not an ETA for the new runner.

The clean Conv3 exp006 surface contains19 injected betas per scheme at initialization
and epoch30. Use only initialization for this stage. Exp007/008 calibrations provide
matched D/F but do not themselves provide EqProp/BPTT cosine at those targets.

### Launch authorization and ownership

User assignment: "launch this; use local gpu, akibscomputer and nom-cool-1 if its
available". The series README holds the amended placement,6.5 GPUh total and
September25 20:00Paris wall deadline; immediate daytime execution is authorized.
Prepare owns scientific configs/runner/calibration; capacity owns staging/launch;
Medium monitor owns observation/collection; root owns analysis/interpretation.
No training is authorized. Record measured commands/hashes/handles below once real.

## Validated completion — September25

All45 target/context matches accepted:27 reused at0.8/2/6 and18 new checks at0.2/10.
No new-point correction was required. New checks took64.59s plus19.22s calibration
parity smoke,0.0233 GPUh. Worst relative matching error across all45 is5.259384e-6
(0.000526 percent); maximum for new points is2.76815e-6. Both new canonical
calibration/smoke bundles validate. All45 lookup source-file hashes were independently
checked; this is calibration evidence, not an H008 cosine outcome.

Lookup: `results/conv123-initial-displacement-noise-map-20260925-v1/beta_lookup/beta_lookup.csv`,
SHA256 `4946df1fd9c62ea94f2bd41e6172b216bba79c9f82b35f05000245b89d270dc3`.
Matched cases are in the sibling `matched_cases.json`. See the result note below.
