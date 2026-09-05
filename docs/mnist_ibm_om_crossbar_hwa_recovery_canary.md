# Exact-P0 HWA recovery canary

This exploratory CUDA study diagnoses the failed ten-epoch standard-crossbar
recovery protocol without programming another array.  It imports exactly the
healthy and published-fault HWA P0 bundles from clean source commit
`1f4ef1b8d05b2b0613bf4e8bec4cce799c034cea`.  The tracked import reference
pins the teacher, HWA master, both P0 bundles, every original manifest/result,
and the complete ancestry chain.

The new execution commit is recorded separately.  Cross-commit input reuse is
allowed only through this canary's explicit verifier; generic completed-run
reuse continues to require the same clean source commit and is not relaxed.

## Coverage

The study contains 30 atomic arms:

- exact healthy/faulted HWA P0;
- hard-label CE/teacher KL;
- LR `0`, `3e-6`, `1e-5`, `3e-5`, `1e-4`, the DRN-q-space match `2e-4`,
  and the prior-grid stress anchor `3e-4`;
- three full 55,000-example epochs, with epoch zero eligible for selection;
- pulse cap 640, expected to be nonbinding but retained as a safety boundary;
- four counterfactual fresh-apparent draws per P0, with no write or mutation.

Each state/LR CE/KL pair stays on one host.  State/rate parity assigns seven
pairs (14 Adam arms) to local CUDA and seven pairs to Akib.  Both matched
fresh-apparent arms run locally with the same configured and runtime-derived
noise stream, so corruption is not confounded with host or random draw.  The
split is 16 disjoint canonical arms locally and 14 on Akib.  LR ranking uses
maximum selected held-apparent accuracy first, selected-minus-own-epoch-zero
declared-objective change second, and lower LR third, with host retained as a
blocking label.

Every arm selects maximum held-apparent validation accuracy, then minimum
declared objective, then the earlier epoch, with epoch zero eligible.  This is
the same scientific checkpoint ordering used for the DRN diagnosis.

Generate and validate the tracked contract before committing:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_analog_relu.generate_hwa_recovery_canary
/home/filip/miniconda3/envs/py312/bin/python \
  -m pytest -q tests/test_mnist_ibm_om_crossbar_hwa_recovery_canary.py
```

Do not launch from a dirty checkout.  Both machines must run the same clean
execution commit.

## Exact imported inputs

Stage this flat directory on both hosts:

```text
data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8/
├── teacher_weights.pt
├── hwa_master.pt
├── hwa_healthy_p0.pt
├── hwa_faulted_p0.pt
└── provenance/
    ├── teacher_manifest.json
    ├── teacher_result.json
    ├── hwa_manifest.json
    ├── hwa_result.json
    ├── healthy_manifest.json
    ├── healthy_result.json
    ├── faulted_manifest.json
    └── faulted_result.json
```

The prepared local cache is
`/tmp/ibm-om-crossbar-hwa-canary-20260905/data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8`.
The declared Akib checkout/cache pair is
`/home/filiposana/staged/ibm_om_crossbar_hwa_canary_20260905` and
`/home/filiposana/staged/ibm_om_crossbar_hwa_canary_20260905/data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8`.

The four staged inputs are byte-pinned as follows:

| Cache file | SHA-256 |
|---|---|
| `teacher_weights.pt` | `7c1f826c6da5e0a8b18024b49305a9c61230a93db28253a2e8ad427d788a4565` |
| `hwa_master.pt` | `b7daa18f5b148114cebc0c6b212099c236b0cebdd588063e72d9e9c127dc6233` |
| `hwa_healthy_p0.pt` | `5e62de40ef1364ce537bd90445a4a9a9c08abc3eb2980b15cd5b5f5144745f0e` |
| `hwa_faulted_p0.pt` | `1530c8424697e3aef7157b20eaa8e6dea3f05e8dabf288d012fae3243adbd809` |

Authenticate it independently on each host:

```bash
python -m experiments.mnist_analog_relu.hwa_recovery_canary verify-inputs \
  --input-dir data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8
```

The verifier loads both device states and checks the internal teacher/HWA
hashes, healthy-to-faulted parent link, assignment/write seeds, and apparent
and persistent tensor hashes.  A filename match alone is never accepted.

## Local shard

First require the exact task interpreter to report CUDA.  The launcher repeats
this check and aborts before preparing a study if it fails.

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_analog_relu.hwa_recovery_canary run \
  --shard local \
  --input-dir data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8 \
  --results-root results \
  --task-python /home/filip/miniconda3/envs/py312/bin/python \
  --aihwkit-python /home/filip/miniconda3/envs/aihwkit/bin/python \
  --mnist-root /home/filip/datasets/mnist
```

Run this command under the local persistent launcher already used for research
jobs.  Do not run it when the CUDA probe reports zero devices.  Complete all
16 authenticated local arms before treating the shard as finished.

## Akib shard

Stage the same clean commit under a new immutable checkout on Akib and copy the
exact old artifacts into its ignored `data/` cache.  Verify the source commit,
input receipt, GPU availability, and occupancy before launch.  The numerical
command is:

```bash
/home/filiposana/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_analog_relu.hwa_recovery_canary run \
  --shard akib \
  --input-dir data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8 \
  --results-root results \
  --task-python /home/filiposana/miniconda3/envs/py312/bin/python \
  --aihwkit-python /home/filiposana/miniconda3/envs/aihwkit-sampler/bin/python \
  --mnist-root /home/filiposana/datasets/mnist
```

Wrap it with the resilient Akib launcher at
`/home/filip/.codex/skills/akib-ssh/scripts/launch_akib_job.sh`:

```bash
/home/filip/.codex/skills/akib-ssh/scripts/launch_akib_job.sh launch \
  --host akib \
  --name ibm_om_hwa_canary_akib \
  --workdir /home/filiposana/staged/ibm_om_crossbar_hwa_canary_20260905 \
  --result-dir /home/filiposana/staged/ibm_om_crossbar_hwa_canary_20260905/launch_records/akib_shard \
  --command '/home/filiposana/miniconda3/envs/py312/bin/python -m experiments.mnist_analog_relu.hwa_recovery_canary run --shard akib --input-dir data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8 --results-root results --task-python /home/filiposana/miniconda3/envs/py312/bin/python --aihwkit-python /home/filiposana/miniconda3/envs/aihwkit-sampler/bin/python --mnist-root /home/filiposana/datasets/mnist'
```

Its result directory is a record only; the canary module continues to own all
canonical study output arguments.  Monitor launcher truth, CUDA activity,
heartbeat, logs, and native artifact growth.  A zero process exit is not
scientific completion without all 14 authenticated Akib arms.

## Collection

Akib is the declared canonical collection host.  Transfer the entire local
materialized study into a new incoming directory on Akib; do not copy files
directly into canonical arm directories.  Then run:

```bash
/home/filiposana/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_analog_relu.hwa_recovery_canary collect \
  --canonical-study-dir /home/filiposana/staged/ibm_om_crossbar_hwa_canary_20260905/results/mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1 \
  --incoming-study-dir /home/filiposana/staged/ibm_om_crossbar_hwa_canary_20260905/incoming/local-study \
  --incoming-shard local \
  --task-python /home/filiposana/miniconda3/envs/py312/bin/python
```

Collection hashes every registered incoming artifact, requires one completion
per assigned arm, requires the same clean execution commit on both hosts, and
copies a whole arm into an empty temporary directory.  It exposes the arm
atomically only after re-verification.  Existing canonical evidence is never
overwritten.  Before copying, it also uses the replicated epoch-zero
evaluations already present in the Adam arms as a cross-host parity sentinel:
student and teacher prediction hashes/counts must match exactly for both P0
states, and held-apparent plus persistent diagnostic metrics must agree within
`1e-5`.  No extra post-collection parity run is required.
It also requires exact per-epoch ordered input/label stream hashes across all
28 Adam arms and authenticates that both local fresh controls used identical
RNG/noise tensors.  A separate pre-production cross-host LR=0 operational
sentinel must gate the full launches; it is not a canonical study
arm and does not replace these collection checks.

After artifact-verified workflow summary reports complete coverage, write the
metric-first diagnostic tables:

```bash
/home/filiposana/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_analog_relu.hwa_recovery_canary analyze \
  --study-dir /home/filiposana/staged/ibm_om_crossbar_hwa_canary_20260905/results/mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1
```

The resulting comparison reports paired epoch-zero, selected, and
training-final held-apparent-primary and persistent-secondary accuracy/CE/KL,
selected epoch, per-epoch/selected/training-final pulse-cap telemetry, and
matched fresh-apparent sensitivity.  The study must go through human
scientific review before it is finalized into the experimental manifest.
