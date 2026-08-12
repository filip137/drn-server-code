# Conv1/2/3 zero-bias long-LR counterfactual

Status: deterministic 18-arm configuration set prepared; no production job is
implied by materialization.

## Question

Can a baseline perfect-diode convolutional network recover the zero-bias SGD
trajectory of ours or legacy merely by changing its layer-specific learning
rates over a full 30-epoch run, and does that answer change with depth?

This is an ordinary-MNIST learning-rate selection and mechanism study. Its
validation metrics are diagnostic and are not paper-facing accuracy evidence.
Any selected vector must be frozen before a separate deterministic
medium-affine confirmation.

## Frozen design

Conv1, Conv2, and Conv3 each contain the same six scientific roles:

1. baseline at its current accepted weight rates;
2. baseline with every Conv rate multiplied by three and Dense unchanged;
3. baseline exactly matched to ours' fixed-cohort initial relative proposals;
4. baseline exactly matched to legacy's fixed-cohort initial relative proposals;
5. ours at its current accepted weight rates; and
6. legacy at its current accepted weight rates.

All 18 arms use ordinary MNIST, seed 0, the canonical architecture-specific
operating point and input gain, `[0,100]` weight projection, plain SGD, exact
zero hidden biases, and 30 epochs. Every run is intended to save initialization
plus all 30 epoch model/optimizer checkpoints and five real optimizer-transition
traces per epoch.

The exact proposal-matched baseline rate for weight `i` is

```text
eta_baseline_to_target_i =
    eta_baseline_i * proposal_target_i / proposal_baseline_i
```

where `proposal_i` is the frozen initialization value of
`LR * RMS(gradient) / RMS(weight)`. The complete source proposal values,
per-parameter multipliers, rates, parent-config hashes, and replay provenance
are embedded in every generated config and in `study_manifest.json`.

The proposal evidence comprises two BPTT-gradient batches from the fixed
128-example replay cohort. The rate derivation consumes no target performance
metric. Exact proposal matching is limited to per-weight initialization RMS
magnitudes on that cohort: it does not match gradient direction, coordinate
distributions, output scaling, the later optimizer trajectory, or establish
global learning-rate optimality.

The frozen measurement sources are recorded by hash even though `results/` is
not part of a staged source archive. Materialization is self-contained; when
the local source artifacts are present, the generator additionally verifies
their bytes.

## Selection

Baseline selection is independent for Conv1, Conv2, and Conv3. For each
architecture, rank only its four baseline candidates by mean validation loss
over epochs 26--30, form an inclusive 2% loss plateau, and apply the fully
declared deterministic tie-breaks in `study_manifest.json`. Ours and legacy
are comparison controls, not selector candidates. Missing, non-finite, or
contract-incomplete arms are ineligible.

This selects among a predeclared mechanistic set; it does not prove a globally
optimal baseline learning rate.

## Prepared Jean Zay envelope

No job is submitted by this package. The planned execution uses
`experiments/run_conv123_zero_bias_lr_counterfactual_jeanzay.slurm` on
`fmu@v100`, `gpu_p13`, `qos_gpu-t3`, `v100-16g`, with one GPU, ten CPUs,
`06:00:00` per task, module `pytorch-gpu/py3/2.5.0`, and array `0-17%6`.
The remote result pattern is
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results/` followed by the
study ID. The conservative budgets are 6, 12, and 24 V100-hours for Conv1,
Conv2, and Conv3 respectively: 42 V100-hours total and roughly seven hours of
ideal compute wall time at concurrency six, plus queueing.

## Conditional paper sequence

Review the SGD mechanism panel before expanding it. If optimizer-general
evidence is needed, first make a new frozen-initialization fresh-state Adam
proposal replay for all three architectures; Adam moment state means that raw
gradient ratios cannot be substituted for optimizer-step proposal ratios.
Only then materialize the analogous Adam counterfactual panel.

After ordinary-MNIST selection, freeze one selected baseline vector per tested
architecture and optimizer. The paper-facing deterministic medium-affine
confirmation uses affine seed 1729, paired model/loader seeds 0, 1, and 2, and
four roles per architecture: baseline current, baseline selected, ours
current, and legacy current. This is 36 arms per optimizer across Conv1/2/3,
with a conservative 84 V100-hour budget per optimizer. Select the best
validation checkpoint, read the official medium-affine test split exactly
once, report paired seed differences and spread, and never retune from the
medium-affine result.

## Materialization

```bash
python -m experiments.prepare_conv123_zero_bias_lr_counterfactual
python -m experiments.prepare_conv123_zero_bias_lr_counterfactual --check
```

The generator reads only the canonical checked-in zero-bias parent configs.
It does not require ignored result files.

## Local preparation gate

On 2026-08-07 the final 18-config set passed a synchronous local CPU smoke
through `experiments.exact_run` with the local MNIST path used only as a
transport override, one real train batch, one validation batch, gradient
tracing, initialization/epoch checkpoints, and the official test disabled.
All 18 canonical bundles validated; their trace and checkpoint inventories
were complete, their split and first-epoch order hashes matched, and every
configured and checkpointed hidden bias remained exactly zero. The ephemeral
smoke root was `/tmp/pd-c123-zb-lr30-smoke-g6TZCT`; it is gate evidence, not a
scientific result directory. The final static suite passed 67 focused tests,
Python compilation, and `bash -n` on the Slurm wrapper.
