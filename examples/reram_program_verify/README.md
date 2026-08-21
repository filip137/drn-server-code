# IBM ReRAM program-and-verify configurations

`smoke_om.json` validates the complete artifact pipeline on a reduced matrix.
It is explicitly tagged `profile: smoke` and cannot satisfy the production
study contract:

```bash
python -m ebl characterize \
  --config examples/reram_program_verify/smoke_om.json \
  --output-dir /tmp/ibm-reram-program-verify-smoke
```

The four `production_*.json` files are the exact arms declared by
`studies/ibm-reram-program-verify-noise-20260821-v1.json`: optimized-material
and baseline-HfO2 presets, each with published corruption disabled and enabled.
Prepare the study before launching any of them:

```bash
python -m ebl study prepare \
  --plan studies/ibm-reram-program-verify-noise-20260821-v1.json
```

Each production config requests 16,121,856 target-programming trajectories
before counting verify-event rows. Expect a large, long-running SQLite artifact;
size storage and execution resources before launch. A completed smoke run is
only an operational gate and is not production evidence.

An OM-continuous sizing pilot on 2026-08-21 used the production pulse and
conditioning caps with 64 identities, two repeats, five targets, one tolerance,
two starts, and two controllers. Its 2,560 trajectories took 36.4 seconds after
matched-conditioning reuse and produced 67,586 verify events in a 5.78-MB
SQLite file. Naive linear scaling is roughly 64 hours and 36 GB of SQLite per
production arm (about 146 GB for four arms), but target, tolerance, preset,
host, and filesystem effects can change both estimates. This pilot is an
operational sizing measurement, not device evidence. Benchmark the target host
and reserve storage before launch; characterization currently preserves a
failed partial database but does not resume it in place.

After preparation, launch each arm into its exact declared study directory,
for example:

```bash
python -m ebl characterize \
  --config examples/reram_program_verify/production_om_continuous.json \
  --output-dir results/ibm-reram-program-verify-noise-20260821-v1/runs/om-continuous
```

Use the corresponding declared config and arm directory for `om-corrupt`,
`hfo2-continuous`, and `hfo2-corrupt`. Do not treat a partial or standalone
run as study coverage.

## Short CUDA production study

The reviewed local production contract is
`studies/ibm-reram-program-verify-noise-20260821-v2.json`. Its four
`production_short_*.json` configs retain both presets, the clean and
published-corrupt populations, all 41 targets, both lower/SET and upper/RESET
starts, both controllers, and identity-held-out analysis. It fixes the
deployment tolerance at the primary `tau/step=0.5`, uses 1,024 identities and
four cycle repeats, and clones one independently sampled boundary state per
device/repeat/start across the target grid. Each arm therefore contains
671,744 trajectories and the complete study contains 2,686,976 trajectories,
24 times fewer than v1.

The CUDA pulse plant uses independently seeded, CPU-generated normal streams
staged on the GPU. AIHWKit population construction and Wan-2022 sampling stay
in the pinned AIHWKit 1.1.0 environment and emit explicit receipts. On the
local host, prepare it with:

```bash
export EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python
/home/filip/miniconda3/envs/py312/bin/python -m ebl study prepare \
  --plan studies/ibm-reram-program-verify-noise-20260821-v2.json
```

The operational gate `smoke_cuda_om.json` completed end to end on the local
RTX 3090 with a passing ledger. The exact-width `sizing_cuda_om.json` gate
then completed 81,920 trajectories over five targets in 61.6 seconds,
including 2,863,651 verify events, full SQLite integrity, endpoint fitting,
Wan comparison, plots, and report generation. Linear target scaling projects
8.4 minutes and about 1.94 GB of SQLite per arm. Even a 10x runtime safety
factor puts four sequential arms below six hours, leaving substantial margin
under the predeclared 24-hour campaign limit. These timing runs are
operational evidence only.

The short study estimates the endpoint model only for `tau/step=0.5`; it does
not support a tolerance-sensitivity claim. The original v1 configs remain
frozen as the superseded exhaustive contract and are not silently redefined.
