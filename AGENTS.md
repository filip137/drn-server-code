# AGENTS — IBM ReRAM program-and-verify noise model

## Worktree scope

These instructions apply to the `codex/ibm-reram-program-verify-noise`
worktree. Its primary research question is:

> Can fixed-amplitude SET/RESET pulse-count program-and-verify over IBM's
> hardware-derived ReRAM array fits yield a validated endpoint noise model,
> and how does that model compare with Wan-2022?

The first milestone in this tree is device characterization, not network
training. It must separate pulse dynamics, controller behavior, endpoint
programming error, read or apparent-state variation, saturation, and corrupt
devices. HWA deployment and Tiki-Taka continuation remain later milestones
and must use the validated model without silently changing device state or
evidence class.

Keep unrelated repository maintenance and unrelated experiment families out
of this worktree.

## Repository map

- `ebl/`: public command-line entry point. Workflow-managed training,
  validation, checkpoint handling, and study lifecycle commands enter through
  `python -m ebl`.
- `model/resistive/`: perfect-diode DRN construction and numerical equations,
  including the passive and digital low-rank adapter structures used by LoRA
  controls.
- `training/`: reusable training mechanisms. The worktree-relevant modules
  include HWA modifiers, measured-trace programming and update backends,
  Tiki-Taka accumulation, parameter catalogs, probes, and checkpoint codecs.
- `experiments/mnist_relu_drn/`: strict teacher-to-perfect-diode-DRN mapping,
  HWA, measured-device deployment, training, validation, and provenance.
- `experiments/mnist_relu_drn_reset/`: measured pulse-zero/RESET
  initialization experiments and their learning-rate protocol.
- `experiments/small_network/`: existing composition root for DRN HWA,
  Tiki-Taka, and low-rank adapter controls when a declared study uses the
  `small_drn.v1` experiment family.
- `examples/mnist_relu_drn/`, `examples/mnist_relu_drn_reset/`, and
  `examples/small_drn/`: strict versioned experiment configs. A study plan
  references exact config files; configs do not choose input checkpoints or
  output locations.
- `studies/`: tracked, predeclared study plans containing hypotheses, arms,
  completion criteria, and analysis plans.
- `campaigns/`: subprocess-only orchestration of exact configs and explicit
  input artifacts, including local and Akib campaign manifests.
- `data/`: ignored staging area for measured synapse data. Its required file,
  digest, and placement are defined in `docs/synapse_data.md`.
- `results/`: ignored raw run directories and generated study analyses. One
  prepared study owns one `results/<study-id>/` root.
- `docs/`: scientific contracts, current synthesis, run ledger, and finalized
  evidence. Protocol documents are authoritative over result summaries.
- `tests/`: strict config, numerical parity, provenance, lifecycle, measured
  update, and study-workflow tests.
- `labs/`: focused reference harnesses for direct perfect-diode and AIHWKit
  Tiki-Taka comparisons. Use them for reference/parity work; workflow-managed
  evidence must use the public `python -m ebl` CLI.

## Architecture

- The evaluated student architecture is a dissipative resistive network (DRN)
  using `perfect_diode` nonlinearities.
- A conventional ReLU network may be used as a frozen teacher or reference,
  but it is not a substitute for the evaluated perfect-diode DRN.
- Matched arms must keep the DRN topology, solver, data split, teacher or base
  checkpoint, and device assignment fixed unless one of them is the explicitly
  declared intervention.

## Synapse data

- Device-facing claims must use measured synaptic data. Synthetic noise models
  and generic device presets may be included only as clearly labelled controls.
- Use the dataset, placement, verification, cohort, and provenance rules in
  [`docs/synapse_data.md`](docs/synapse_data.md).
- Do not silently replace the measured dataset, alter its preprocessing, or
  mix device cohorts between arms.

## Initialization

- Every study must predeclare its initialization path and follow
  [`docs/initialization_protocols.md`](docs/initialization_protocols.md).
- HWA and non-HWA arms must start from the same logical state. Recovery arms
  must start from the same explicitly named deployed state unless
  initialization itself is the intervention.
- Never discover or substitute the newest checkpoint. Record explicit input
  paths and content hashes.

## Experimental workflow

- Follow [`docs/experiment_workflow.md`](docs/experiment_workflow.md).
- Declare the hypothesis, arms, completion criteria, and analysis before
  launching native runs.
- Keep workflow-managed raw runs under the prepared
  `results/<study-id>/runs/<arm-id>/` roots and preserve failed attempts.
- Separate measured results from interpretation. Claim that on-chip training
  is needed only when the predeclared HWA-only control underperforms and a
  matched on-chip arm closes the specified gap.
