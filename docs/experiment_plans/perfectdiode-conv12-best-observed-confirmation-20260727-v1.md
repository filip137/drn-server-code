# Perfect-Diode Conv1/Conv2 Best-Observed Confirmation

Plan lifecycle status: `approved_preflight_required`.

This is a diagnostic-only successor to the completed ordinary-MNIST rho
screens. The selected cells are best-observed overrides, not protocol-frozen
learning rates. Filip explicitly re-approved the unchanged twelve-entry
experiment for Jean Zay on 2026-07-27, with two concurrent runs per allocated
V100. The contract below freezes six fixed two-run Slurm packs and binds the
R3 allocation, reviewed config, generated manifest, clean source archive, and
content-addressed staging paths.

```json
{
  "schema_version": "experiment-run-plan/v1",
  "experiment_id": "perfectdiode-conv12-best-observed-confirmation-20260727-v1",
  "title": "Perfect-Diode Conv1/Conv2 Best-Observed Confirmation",
  "evidence_scope": "diagnostic",
  "hypothesis": {
    "statement": "The twelve prespecified best-observed perfect-diode rho cells can train from their architecture-shared seed-0 initializations for the requested longer horizons at the unchanged ordinary-MNIST operating points, while remaining numerically finite, safety-admissible, and exactly reproducible.",
    "control": "For each entry, the hash-verified safety-admissible three-epoch candidate named in the canonical config supplies the rho target and raw parameter-specific learning-rate vector; every confirmation restarts from the same architecture-shared seed-0 initialization rather than continuing from that candidate checkpoint.",
    "treatments": [
      "Conv1 baseline SGD at rho_conv=0.009 and rho_dense=0.03 for 10 epochs",
      "Conv1 baseline Adam at rho_conv=0.009 and rho_dense=0.03 for 10 epochs",
      "Conv1 ours SGD at rho_conv=0.009 and rho_dense=0.03 for 10 epochs",
      "Conv1 ours Adam at rho_conv=0.009 and rho_dense=0.03 for 10 epochs",
      "Conv1 legacy SGD at rho_conv=0.003 and rho_dense=0.03 for 10 epochs",
      "Conv1 legacy Adam at rho_conv=0.003 and rho_dense=0.01 for 10 epochs",
      "Conv2 baseline SGD at rho_conv=0.027 and rho_dense=0.09 for 20 epochs",
      "Conv2 baseline Adam at rho_conv=0.027 and rho_dense=0.09 for 20 epochs",
      "Conv2 ours SGD at rho_conv=0.009 and rho_dense=0.03 for 20 epochs",
      "Conv2 ours Adam at rho_conv=0.027 and rho_dense=0.03 for 20 epochs",
      "Conv2 legacy SGD at rho_conv=0.003 and rho_dense=0.03 for 20 epochs",
      "Conv2 legacy Adam at rho_conv=0.009 and rho_dense=0.01 for 20 epochs"
    ],
    "expected_direction": "Each entry is expected to complete its exact optimizer-step budget with finite loss, gradients, updates, states, checkpoints, validation metrics, and safety diagnostics; no cross-scheme or cross-optimizer ordering is hypothesized.",
    "decision_rule": {
      "metric": "Exact completion plus per-epoch validation loss, validation accuracy, and the bound safety diagnostics, interpreted independently for every entry.",
      "split": "The hash-bound ordinary-MNIST seed-0 training order and stratified 5000-example validation split; the official test set must not be read.",
      "checkpoint_role": "Best-validation-loss and final checkpoints from a restart at the architecture-shared initialization after exactly 34380 Conv1 steps or 68760 Conv2 steps.",
      "seed_aggregation": "One prespecified model and shuffle seed, seed 0; no cross-seed aggregation.",
      "comparison": "Compare each longer confirmation only with its own hash-bound three-epoch source candidate and report all twelve entries separately; do not infer cross-scheme superiority or reinterpret the selected rho values as frozen protocol handoffs.",
      "minimum_meaningful_effect": 0.0,
      "effect_units": "This is an exact-completion and trajectory diagnostic; validation changes are reported descriptively rather than thresholded as an efficacy effect."
    },
    "supports_if": "All twelve entries complete their exact epoch and step budgets from the bound shared initializations, publish hash-verified best and final checkpoints and per-epoch validation metrics, remain safety-admissible, and preserve official_test_read=false.",
    "does_not_support_if": "One or more entries reaches its full budget but has a structured numerical or safety failure, or the longer trajectory cannot be reproduced from the bound initialization and parameter-specific learning-rate vector.",
    "inconclusive_if": "Any required entry, source candidate, probe, initialization, split, fresh T/K gate, source identity, environment identity, completion marker, checkpoint, metric, Slurm task state, transfer receipt, or artifact hash is missing or unverifiable."
  },
  "sweep": {
    "config_path": "configs/conv/perfectdiode_conv12_best_observed_confirmation_20260727_v1.json",
    "config_sha256": "3635c5866b27445c58c90de034c4f2e152919772b40258820f614a60a61ea740",
    "manifest_path": "results/.launch_staging/perfectdiode-conv12-best-observed-confirmation-20260727-v1/pdconfirmbundle_c56239abf6f84bc74fdc0ccb5c0054371cd43ad66631f4b3ce2e61d094568e5d/manifest.json",
    "manifest_sha256": "63bf6496d7e790fda7a5e8c3f8a677a25263c9aded65ac2eb373c51b707b7f61",
    "manifest_job_count_pointer": "/entries",
    "axes": [
      {
        "name": "ordered_confirmation_entry",
        "config_path": "/entries",
        "ordered_values": [
          "pdconfirm_conv1_baseline_sgd_seed0",
          "pdconfirm_conv1_baseline_adam_seed0",
          "pdconfirm_conv1_ours_sgd_seed0",
          "pdconfirm_conv1_ours_adam_seed0",
          "pdconfirm_conv1_legacy_sgd_seed0",
          "pdconfirm_conv1_legacy_adam_seed0",
          "pdconfirm_conv2_baseline_sgd_seed0",
          "pdconfirm_conv2_baseline_adam_seed0",
          "pdconfirm_conv2_ours_sgd_seed0",
          "pdconfirm_conv2_ours_adam_seed0",
          "pdconfirm_conv2_legacy_sgd_seed0",
          "pdconfirm_conv2_legacy_adam_seed0"
        ]
      }
    ],
    "case_ids": [
      "pdconfirm_conv1_baseline_sgd_seed0",
      "pdconfirm_conv1_baseline_adam_seed0",
      "pdconfirm_conv1_ours_sgd_seed0",
      "pdconfirm_conv1_ours_adam_seed0",
      "pdconfirm_conv1_legacy_sgd_seed0",
      "pdconfirm_conv1_legacy_adam_seed0",
      "pdconfirm_conv2_baseline_sgd_seed0",
      "pdconfirm_conv2_baseline_adam_seed0",
      "pdconfirm_conv2_ours_sgd_seed0",
      "pdconfirm_conv2_ours_adam_seed0",
      "pdconfirm_conv2_legacy_sgd_seed0",
      "pdconfirm_conv2_legacy_adam_seed0"
    ],
    "seeds": [
      0
    ],
    "conditional_expansion": {
      "enabled": false,
      "rule": null,
      "maximum_additional_jobs": 0
    },
    "expected_initial_job_count": 12,
    "maximum_total_job_count": 12,
    "undeclared_fields_fixed_by": [
      "configs/conv/perfectdiode_conv12_best_observed_confirmation_20260727_v1.json",
      "configs/conv/perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json",
      "configs/conv/perfectdiode_conv2_high_rho_corner_20260727_v1.json",
      "configs/conv/perfectdiode_successor_jeanzay_fmu_v100_environment_v1.json",
      "docs/perfectdiode_learning_protocol.md"
    ]
  },
  "execution": {
    "launcher": [
      "/lustre/fshomisc/sup/hpe/pub/miniforge/24.9.0/envs/pytorch-gpu-2.5.0+py3.12.7/bin/python",
      "/lustre/fswork/projects/rech/umg/ucy17uy/server_code/staged_sources/perfectdiode-conv12-best-observed-confirmation-20260727-v1/228d17636e036e49e0e16302f52bcf7f85efb4c150d8786220eb204e9d9e310e/source/experiments/supervise_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py",
      "--state",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1/preflight/attempt-06-r3/supervisor-state.json",
      "--bundle-dir",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/.launch_staging/perfectdiode-conv12-best-observed-confirmation-20260727-v1/pdconfirmbundle_c56239abf6f84bc74fdc0ccb5c0054371cd43ad66631f4b3ce2e61d094568e5d",
      "--output-root",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1",
      "--data-root",
      "/lustre/fsn1/projects/rech/umg/ucy17uy/datasets/mnist",
      "--source-archive",
      "/lustre/fswork/projects/rech/umg/ucy17uy/server_code/staged_sources/perfectdiode-conv12-best-observed-confirmation-20260727-v1/228d17636e036e49e0e16302f52bcf7f85efb4c150d8786220eb204e9d9e310e/source.tar.gz",
      "--environment-contract",
      "/lustre/fswork/projects/rech/umg/ucy17uy/server_code/staged_sources/perfectdiode-conv12-best-observed-confirmation-20260727-v1/228d17636e036e49e0e16302f52bcf7f85efb4c150d8786220eb204e9d9e310e/source/configs/conv/perfectdiode_successor_jeanzay_fmu_v100_environment_v1.json",
      "--launch-authorization-receipt",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1/preflight/attempt-06-r3/launch-authorization.json",
      "--launch-authorization-receipt-sha256",
      "__PD_SUCCESSOR_LAUNCH_AUTHORIZATION_SHA256__",
      "--scheduled-preflight-receipt",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1/preflight/attempt-06-r3/scheduled-run-preflight.json",
      "--preflight-receipt",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1/preflight/attempt-06-r3/successor-preflight.json",
      "--official-canary-receipt",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1/preflight/attempt-06-r3/official-canary-receipt.json",
      "--canary-receipt",
      "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1/preflight/attempt-06-r3/successor-canary-gate.json",
      "--repo-root",
      "/lustre/fswork/projects/rech/umg/ucy17uy/server_code/staged_sources/perfectdiode-conv12-best-observed-confirmation-20260727-v1/228d17636e036e49e0e16302f52bcf7f85efb4c150d8786220eb204e9d9e310e/source",
      "--remote-user",
      "ucy17uy",
      "--official-verifier",
      "/lustre/fswork/projects/rech/umg/ucy17uy/server_code/launch_tools/verify_canary.py",
      "--canary-pack-index",
      "4",
      "--poll-seconds",
      "60",
      "--enable-submit"
    ],
    "collector": [
      "python",
      "experiments/run_mnist_conv_perfectdiode_successor_confirmation.py",
      "status",
      "--bundle-dir",
      "results/.launch_staging/perfectdiode-conv12-best-observed-confirmation-20260727-v1/pdconfirmbundle_c56239abf6f84bc74fdc0ccb5c0054371cd43ad66631f4b3ce2e61d094568e5d",
      "--output-root",
      "results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1"
    ],
    "validator": [
      "python",
      "experiments/run_mnist_conv_perfectdiode_successor_confirmation.py",
      "validate-bundle",
      "--bundle-dir",
      "results/.launch_staging/perfectdiode-conv12-best-observed-confirmation-20260727-v1/pdconfirmbundle_c56239abf6f84bc74fdc0ccb5c0054371cd43ad66631f4b3ce2e61d094568e5d"
    ],
    "preflight": {
      "smoke_required": true,
      "tk_reference_required": true,
      "scheduled_run_preflight_required": true,
      "receipt_path": "results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1/preflight/attempt-06-r3/scheduled-run-preflight.json"
    }
  },
  "storage": {
    "root_alias": "REPO_ROOT",
    "local_results_root": "results",
    "local_bundle_path": "results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1",
    "remote_staging": [
      {
        "host": "jean-zay",
        "path": "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1",
        "intended_local_destination": "results/perfectdiode_best_observed_confirmation_studies/perfectdiode-conv12-best-observed-confirmation-20260727-v1"
      }
    ]
  },
  "reporting": {
    "progress_tracker": "docs/current_experiments.md",
    "comparison_cards": [
      {
        "comparison_id": "perfectdiode-conv1-best-observed-10epoch-confirmation-20260727-v1",
        "card_schema_version": "drn-result-card/v1",
        "card_path": "result_registry/cards/diagnostics/perfectdiode-conv1-best-observed-10epoch-confirmation-20260727-v1.json",
        "review_path": "result_registry/reviews/perfectdiode-conv1-best-observed-10epoch-confirmation-20260727-v1.json",
        "final_results_anchor": "docs/results/index.md#perfectdiode-conv1-best-observed-10epoch-confirmation-20260727-v1"
      },
      {
        "comparison_id": "perfectdiode-conv2-best-observed-20epoch-confirmation-20260727-v1",
        "card_schema_version": "drn-result-card/v1",
        "card_path": "result_registry/cards/diagnostics/perfectdiode-conv2-best-observed-20epoch-confirmation-20260727-v1.json",
        "review_path": "result_registry/reviews/perfectdiode-conv2-best-observed-20epoch-confirmation-20260727-v1.json",
        "final_results_anchor": "docs/results/index.md#perfectdiode-conv2-best-observed-20epoch-confirmation-20260727-v1"
      }
    ],
    "final_results_page": "docs/results/index.md"
  },
  "completion": {
    "required_coverage": "Exactly twelve hash-verified production outcomes from six fixed two-run Slurm packs: six Conv1 confirmations completing 10 epochs and 34380 optimizer steps each, and six Conv2 confirmations completing 20 epochs and 68760 optimizer steps each; one passing same-path two-process live canary for entries 8 and 9; fresh passing T/K reference gates for all six architecture-by-scheme rows; terminal Slurm parent and task states; a complete locally repatriated and validated bundle; and two reviewed architecture-specific diagnostic cards.",
    "allowed_exclusions": [
      "No production entry may be silently omitted, substituted, retried from a candidate checkpoint, or assigned a different rho or learning-rate vector.",
      "A failed entry remains a structured diagnostic failure; it does not authorize a replacement cell or conditional expansion.",
      "No official-test measurement, paper-facing evidence, frozen learning-rate handoff, or cross-scheme superiority claim is allowed."
    ],
    "failure_handling": "Fail closed on any approval, plan, config, source, environment, parent artifact, initialization, split, probe, rho, raw learning-rate vector, T/K gate, manifest, Slurm state, completion, checkpoint, output, transfer, or hash mismatch. Resume only hash-verified complete entries; restart every incomplete entry from its architecture-shared seed-0 initialization.",
    "local_validation_required": true,
    "review_required": true
  },
  "approval": {
    "status": "approved",
    "approved_by": "Filip",
    "approved_at": "2026-07-27T23:38:34+02:00",
    "amendment_of": "attempt-05-r3"
  }
}
```

## Exact approval and packing amendment

Filip explicitly approved this experiment ID and its scientific contract on
2026-07-27, then specified one GPU for every two runs. Repository precedent
defines that request as two concurrent ordinary processes sharing one
allocated GPU. This pre-launch execution amendment creates six fixed packs
`[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`, `[8,9]`, and `[10,11]`; it does not change
the twelve-entry manifest or any rho, epoch, seed, initialization, dataset, or
`T/K` choice.

The attempt-05 implementation must pass the local protocol and lifecycle test
gate, after which its config, manifest, source, environment, plan, and bundle
identities are frozen. Launch remains fail-closed until the scheduled-run
preflight and the live paired canary both pass.

The first live `sbatch --test-only` correctly failed closed because Jean Zay
forbids `--mem`, `--mem-per-cpu`, and `--mem-per-gpu`. The execution profile
therefore records the site-managed host-memory policy and emits no explicit
Slurm memory option. This operational correction does not change the GPU,
CPU, packing, walltime, scientific entries, or any training parameter, and
all prior preflight identities and receipts were invalidated before retry.

Attempt-02 live canary job `311193` then failed closed before any training
process started because Jean Zay reports `NumCPUs=32` after expanding the
requested 16 physical CPUs into logical CPUs on its two-thread V100 nodes.
The attempt-03 audit now independently requires `CPUs/Task=16`,
`ReqTRES cpu=16`, one task, one node, and one thread per core, while accepting
only Jean Zay's pending `NumCPUs=16` or allocated `NumCPUs=32` accounting
representation. No attempt-02 payload artifact exists, and its failed job,
source, bundle, authorization, and receipts remain immutable provenance.

Attempt-03 live canary job `311705` received and verified one V100-32GB, then
failed before any training process while recomputing the source-archive
fingerprint: the 116 MiB archive expands to about 214 MiB, but Python's
temporary extraction defaulted to Jean Zay's size-limited compute-node
`/tmp`. Attempt-04 follows the IDRIS batch-storage contract by requiring the
job-unique, writable `JOBSCRATCH` directory and exporting it as `TMPDIR` only
on the live worker path. This preserves the archive SHA, extracted-source
fingerprint, and staged-checkout fingerprint checks; it changes no scientific
or GPU execution parameter. Job `311705` created no training artifact, and
production remained blocked.

Filip subsequently re-authorized these same runs on Jean Zay and requested
automatic production release after every required gate passes. Attempt-05-r3
invalidates every earlier Jean Zay and Trex preflight identity and changes
only the operational allocation and result ownership: `fmu@v100` on
`gpu_p13` with `qos_gpu-t3`, one V100-32GB per fixed two-process pack, and
FMU-owned result paths. The immutable source and MNIST dataset remain on the
existing UMG-owned shared read paths. This attempt binds clean source commit
`367abf8d0f9a76b562a65a0b80c0ebbaef6624dc`, source archive SHA-256
`228d17636e036e49e0e16302f52bcf7f85efb4c150d8786220eb204e9d9e310e`,
effective code fingerprint
`8d2631c841a5ddaf571060a50cd42a9ecff511ba60e900afcc179a5b9d5d5958`,
and immutable bundle
`pdconfirmbundle_c56239abf6f84bc74fdc0ccb5c0054371cd43ad66631f4b3ce2e61d094568e5d`.
It requires a fresh scheduled-runner receipt, all six fixed-T/K security
replays, scheduler test-only acceptance, and the same concurrent full-epoch
entries 8 and 9 live canary before production. No rho, raw learning-rate
vector, epoch budget, seed, initialization, dataset split, or `T/K` value
changes.

Attempt-06-r3 is the receipt-only successor authorized by Filip's instruction
to repair launch-contract errors and continue automatically. It preserves the
attempt-05-r3 plan-validator failure and reuses the identical immutable source,
bundle, config, environment, and scientific contract; only fresh receipt paths
and the approval-to-canary clock are amended.

The recorded approval accepts:

- these twelve source cells and raw learning-rate vectors as
  diagnostic-only best-observed overrides;
- Conv1 at exactly 10 epochs and 34,380 steps;
- Conv2 at exactly 20 epochs and 68,760 steps;
- unchanged Conv1 `T/K=4/4` and Conv2 `T/K=6/6`;
- restart from the shared seed-0 initialization with no candidate-checkpoint
  continuation;
- Jean Zay dossier `AD010913993R3`, account `fmu@v100`, partition
  `gpu_p13`, QoS `qos_gpu-t3`, and constraint `v100-32g`;
- six fixed two-run GPU packs, with the heaviest Conv2 pair `[8,9]` used for
  the live concurrent canary; and
- the fail-closed local, scheduled, scheduler, fixed-T/K, live canary,
  transfer, and local-validation gates.

Once the final approval record and hashes are frozen, it authorizes only
implementation-complete preflight and, if every gate passes, the exact
singleton packed canary and six production packs yielding the twelve entries
above. It does not authorize any other rho, epoch budget,
`T/K`, dataset, seed, host, official-test read, paper-facing claim, or
replacement run.
