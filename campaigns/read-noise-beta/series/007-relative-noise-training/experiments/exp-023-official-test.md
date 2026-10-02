---
id: "exp-023"
title: "Sealed official-MNIST tests for read-noise comparisons"
status: "complete"
hypotheses: []
---
# Exp023 — Official-test evaluation of saved read-noise models

## Authorization and scope

On October 1 Filip requested a list of all needed test results and immediate
execution, using the largest evaluation batch that fits on local, Akib's computer
and nom-cool-1. This is checkpoint evaluation only, without training or retuning.
The adopted scope is the153-case read-noise paper comparison at seed-0-reference D/F=1/4/6. The complete
inclusion set and selected checkpoint hashes must be sealed before test access.

Sealed main grid: Conv1/2/3 × baseline/ours/legacy, comprising 27 clean cases
(seeds0/1/2), 54 noisy seed0 cases (eta1e-6,1e-5,1e-4,3e-4,5e-4,1e-3), and
72 noisy seeds1/2 cases at eta1e-5,1e-4,3e-4,5e-4: **153 evaluations**.
Source studies exp014–019 and exp022 own training provenance. Tiny-eta historical
cases without matching baseline/deep-model horizons are outside this main surface.
Batch128 exp021 and higher-current exp020 remain separate unless requested.

## Frozen scientific choices

Apply the existing paper checkpoint rule: highest validation accuracy over the
complete declared training history, earliest epoch on a tie. Horizons10/30/30.
For historical continuations, consider both parent epochs1–10 and continuation
11–30 and retain the Adam/RNG reset label. New seeds are uninterrupted; never
pool reset trajectories into their primary means. Record selected epochs and
hashes before looking at official test results. This differs from exp022's
fixed-final-epoch validation primary and must be labeled in comparisons.

Use the saved configuration's preprocessing, architecture, amplification, perfect
diodes, native inference iterations4/6/8, float64 and zero biases. Evaluation is
clean and resets state per batch. Evaluate all10000 official MNIST test examples
exactly once per selected model; report accuracy, correct count and loss. No
selection, exclusions or tuning based on test outcomes. Freeze source, full
inclusion set, config/checkpoint identity and reuse audit before evaluation.

Batch sizing uses validation images only. Search the largest safe integer batch
within10000 with at least1GiB/10% GPU memory headroom, verify the real forward path,
and freeze each execution batch before test access. Preserve an exclusive durable
checkpoint test-access marker; a partial official test is recorded and escalated,
never silently rerun. GPU placement is transport, not a change of scientific config.

## Budget and evidence

Result root: `results/read-noise-official-test-20261001-v1/`, including all smoke,
probe, failed, remote-collected and canonical evaluation artifacts. Case inventory,
sealed contract and reuse/source audits live below preparation/. Expected work is
153 single-pass evaluations plus validation-only batching probes; initial ceiling
12 GPU-hours across permitted hosts, 24-hour deadline from first GPU probe.
Measure throughput before assigning final per-worker limits; do not exceed this
budget or add training. Anticipated wall time is tens of minutes to a few hours,
subject to measured inference throughput.

Expected result note: `../results/exp-023-official-test.md`. Report every expected
case, failures and test-access counts. Summaries separate uninterrupted seeds1/2,
compatible supplementary seed0, clean controls and historical reset trajectories.
No statistical-significance claim from these small seed counts.

## Execution and monitoring handoff

Owner: root Reviewer; all153 evaluations collected, validated and reviewed. No outstanding study polling or collection remains.
Targets verified October1: local nom-cool-2 RTX3090 24GiB, nom-cool-1 RTX3090
24GiB, Akib RTX3080 10GiB. nom-cool-1 has Ben's idle MPS process; preserve it.
The existing shared run-watch service is alive and will own submitted jobs;
record exact queue IDs, commands, source identities, batch receipts and verified
service observations here before relinquishing supervision.

### Frozen inclusion and launch preparation

All153 selected checkpoints hash-match their saved artifacts and all153 prior
training metrics report zero official-test evaluations/examples. There are20
historical reset trajectories. [Full case list](../../../../../results/read-noise-official-test-20261001-v1/preparation/test-list.csv)
records every model, scheme, eta, seed, selected epoch and optimizer regime.

Sealed contract SHA256:
`907cf036aafb3bebefe7ef76ed4490e85cc0a7841165f313b8e18275d225da43`.
The417-file source snapshot is in source/ and every file hash is in contract.json.
Nine historical source roots were checked: only default-target-scale plumbing
and checkpoint/RNG lifecycle code differ; free inference is unchanged. The helper
now converts precision before strict checkpoint loading and verifies exact saved
parameter values, avoiding any float32 round trip. Fourteen CPU guard tests pass.

Placement is sealed:51 Conv1 cases on Akib,51 Conv2 locally,51 Conv3 on nom-cool-1.
Three validation capacity profiles per host (one per scheme) cover all17 cases
per profile. Selected-batch predictions/scores are checked against batch64 on a
fixed128-image validation cohort. Inference is clean and float64 throughout.

Shared queue accepted `exp023-probe-local-20261001-v1`. It owns this local probe;
root retains preparation and subsequent test-submission ownership. All prepared
probe/test job upper bounds total7.5GPUh, below12GPUh cap;24h deadline is in
transport/window.json. Remote staging is pending explicit checkpoint-transfer
approval: automatic approval review rejected the Akib rsync as model-weight
export, despite the authorized use of that GPU. No remote copy or test has run.

### Local capacity passed; official test submitted

All153 sealed config/checkpoint/source/reuse guards pass without dataset access.
The three local Conv2 validation probes completed with batch10000, peak11452MiB,
2300MiB headroom, repeated endpoint success and matching128-example predictions
against batch64 (scores within1e-9 relative/1e-11 absolute tolerance). Thus each
Conv2 model evaluates the entire official test set in one batch. Probe evidence:
workers/probe-local-v1/. Queue `exp023-test-local-20261001-v1` accepted51 local
evaluations, expected about15min with7200s maximum; admission uses measured
11452MiB peak plus2300MiB headroom. No automatic test retry is permitted.

Remote102 evaluations remain unlaunched pending explicit checkpoint-transfer
approval. Local and remote group validators require canonical artifacts for
every expected case and exact10000-example once-only completion. The queue
owns local process/artifact checks and validation; root owns remote staging and
aggregate scientific review. No separate poller is attached to these jobs.

Startup verified: local test supervisor PID2274609 and worker PID2274624 match
the sealed command. Three cases have canonical complete results, each one batch
of10000 official test examples and exactly one evaluation; the group advanced
to its fourth case. This is semantic progress beyond the initial launcher check.
The queue service remains the sole local monitoring owner, on its60s cadence.

### Explicit remote transfer approval

Filip explicitly approved copying the selected checkpoint/config files to both
Akib and nom-cool-1. This resolves the automatic-review export block above.
Root is staging the unchanged sealed source/assets and will submit their
validation-only capacity probes before official testing; no scientific choice
or checkpoint identity changes.

Both remote asset/source transfers completed with exit0. Original remote probes
stopped before dataset construction because staged source directories lacked Git
metadata. Root initialized real Git snapshots of the exact staged Python files;
contract and417 source hashes remain unchanged. Runtime snapshot commits are
recorded in transport/remote-provenance-repair.json. Failed probe-v1 artifacts
remain preserved. Queue replacements `exp023-probe-akib-20261001-v2` and
`exp023-probe-nom-cool-1-20261001-v2` own capacity verification. Their test jobs
consume probe-v2 receipts; local remains probe-v1. Prepared upper bounds including
both replacement probes total8.5GPUh, still below12GPUh; charge actual failed
allocation time too. No official test was consumed by these failures.

Akib probe-v2 passed all three Conv1 schemes with batch10000,6154MiB peak and
1024MiB headroom. `exp023-test-akib-20261001-v1` has been submitted with
automatic collection and local validation. Local51/51 official tests completed
and queue validation passed (499.954seconds charged); no failures or repeated
test access. Remote test completion hooks safely synchronize only worker and
test-access evidence, retaining remote artifacts and historical probe failures.

Independent local collector audit passed51/153 expected test cases with zero
validation errors. It checked canonical source/config/checkpoint identities,
exact10000-example counts,51 unique durable test reservations and raw dataset
identity. analysis/summary.json and coverage.csv retain the pending102 cases.
The remote completion hooks run analysis/collect.py and regenerate descriptive
tables/JPGs without executing models or reopening test datasets.

### All production groups submitted

All nine validation capacity profiles passed at batch10000 with matching
batch64 predictions. Peak reserved memory: Conv1/Akib6154MiB, Conv2/local11452MiB,
Conv3/nom-cool-1 21024MiB; headroom1024/2300/2300MiB respectively. The full
10000-example official test is thus one batch per model on each assigned GPU.

Queue `exp023-test-nom-cool-1-20261001-v1` now owns the final51 cases, expected
about25min with7200s maximum. Both remote production jobs have controller-side
collection hooks with600s timeout and safe collection-only retries. Seven bounded
collector semantic checks pass, including rejection of missing examples, repeat
access, changed sealed identity and duplicate reservations. Scientific test
execution remains nonretryable. No extra training or test cells are introduced.

Final strict audit after collection:
`/home/filip/miniconda3/envs/py312/bin/python
results/read-noise-official-test-20261001-v1/analysis/analyze.py`.
It requires153 valid cases and all target groups complete; partial means retain
their actual n. The reviewer must then close the owning result note and ledger.

### Conv3 missing-dataset failure before any test read

The first nom-cool-1 production attempt stopped in torchvision's MNIST constructor:
only train files were present, and `_check_exists()` raised before `_load_data()`.
No legacy processed dataset existed; both official raw files were absent. The
worker exited, metrics were empty and no dataset identity/result existed. The
runner had conservatively reserved access before constructing the loader. Root
review proved zero test reads/evaluations; its unused reservation was archived
under test-access/aborted-before-read/ on remote and local copies, never deleted.
All original failure artifacts remain immutable. Ten evidence-file hashes, the
actual loader source and missing-file observation are recorded in
transport/test-attempt-reconciliation.json and nom1-dataset-pre-read-evidence.json.

The public official MNIST files were copied from local to nom-cool-1. Recovery
uses test-nom-cool-1-v2 with unchanged scientific source/config/checkpoint hashes,
all51 original cases and batch10000. The collector requires the explicit strong
no-read proof before accepting this replacement; it never ignores partial test
access. Including this replacement's full allowance, prepared upper bounds are
10.5GPUh, below12GPUh; actual failed allocation time remains charged.

Akib51/51 tests completed and the automatic collection/validation hook succeeded.
Combined local+Akib coverage is102/153 with zero audit errors and identical raw
test dataset hashes. The remaining51 are all Conv3; no extra cases are authorized.

Independent collector audit accepted all10 no-read proof hashes and classified
the failed production attempt as superseded_zero_read. Current audit remains
102/153 valid with zero errors; all51 Conv3 cases remain pending in the v2
coverage until measured. Replacement queue job
`exp023-test-nom-cool-1-20261001-v2` was submitted after that audit passed.

Six reconciliation regression checks pass, including rejection of nonempty failed
metrics, nonzero test counts, possible legacy-data access and missing proof hashes.
The collector and analyzer share this fail-closed attempt mapping. Frozen source
and all153 selected checkpoint identities remain unchanged across recovery.

Final startup handoff: replacement nom-cool-1 worker is alive on its assigned
RTX3090 and completed the first Conv3 case with exactly10000 examples, one test
evaluation, zero optimizer steps and unchanged parameters; it advanced to seed1.
The persistent shared queue remains sole monitoring owner and will collect/
validate the complete51-case group automatically. Local+Akib102 cases are already
collected/validated. Receipt: transport/remote-ownership-receipt.json. Root owns
the final scientific review after strict153-case coverage passes.

### Completed and reviewed

All153 official-test cases and all three production groups are complete. The final
worker ended2026-10-01T12:39:44.776413Z (14:39Paris); automatic remote collection
and validation succeeded. Strict analysis passed153/153, zero pending and zero
errors, with153 actual test evaluations and one separately audited zero-read
reservation. Measured worker charge including probes/failures:0.722705GPUh.

[Reviewed result](../results/exp-023-official-test.md) owns the final measurements,
interpretation and paper-use limits. Source/checkpoint/inclusion identity remained
frozen; no test retuning or manuscript edit occurred. The shared queue retains
old failed attempts as terminal history; no study worker or collection is active.
