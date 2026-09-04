# AGENTS — IBM OM cell-aware quantized exploration

## Worktree scope

These instructions apply to the `codex/ibm-om-cell-aware-quantized` worktree.
Its current research question is:

> How should the IBM optimized-material conductance baseline be shared and
> placed, together with the quantization spacing, when cells have different
> commissioned RESET-to-SET windows?

Work in this tree must separate ideal logical mapping from persistent physical
programming, and must keep baseline sharing, baseline position, spacing, and
device assignment distinguishable in the results. The immediate focus is to
choose between a baseline shared by four cells and one shared by two cells,
then determine where that baseline should lie given the cells' asymmetric
headroom and the requested spacing.

Keep unrelated repository maintenance and unrelated experiment families out
of this worktree.

## Mandatory DRN apparent-state forwards

- Whenever a device model distinguishes a hidden persistent state from a
  post-write apparent state, **every device-facing DRN forward pass must use
  the current held apparent state**. This includes forwards used for off-chip
  HWA, on-chip recovery, gradient computation, training metrics, validation,
  checkpoint selection, and final evaluation. Never substitute the hidden
  persistent state in any of those paths.
- Pulses and physical updates mutate the persistent state. After each write,
  refresh the apparent state of touched cells according to the declared device
  and write-noise model; the next DRN forward uses that updated held apparent
  state. Do not copy apparent values into persistent state, and do not redraw
  write noise independently per example unless a separately declared
  inference-read-noise intervention requires it.
- A persistent-state DRN forward is permitted only as an explicitly named
  diagnostic or ablation. It must not drive gradients, optimizer updates,
  checkpoint selection, or headline conclusions. When both states exist,
  report apparent-state accuracy as primary and persistent-state accuracy as a
  clearly labelled secondary diagnostic on the same examples and settings.
- For off-chip HWA without an evolving physical plant, use the sampled
  noisy/apparent hardware view for the forward and update the digital master
  or desired target. Do not describe that digital-master update as a
  persistent-device update.

## Exploratory operating mode

This is an exploratory worktree. The default objective is to obtain a
scientifically interpretable result as soon as practical, rather than to
reproduce production CI, release gates, or finalized-study ceremony on every
iteration.

- Prefer the shortest end-to-end path that answers the current question. Use
  direct local CUDA execution, focused scripts, reduced assignments or
  repetitions, and staged sweeps when they provide faster feedback.
- Run only the smallest smoke needed to catch likely errors in the changed
  path: normally an import or config check plus one tiny numerical or GPU
  canary. Documentation-only changes need no runtime smoke.
- Do not run broad test suites, exhaustive CPU/GPU parity, duplicate execution
  surfaces, multi-platform checks, production-launcher validation, lifecycle
  checks, or repository-wide security scanners unless the change or the user
  request makes them directly relevant.
- Repository security and hardening checks are targeted, not automatic. Run
  them when changing credentials, network access, subprocess execution,
  deserialization, permissions, untrusted-input handling, or artifact
  integrity. Do not weaken production security code to make an experiment
  pass.
- “Loosen security checks” means skipping unrelated repository scanners and
  release gates. It never means bypassing sandbox or approval controls,
  exposing credentials, weakening access control, or taking unsafe destructive
  actions.
- Exploratory runs may bypass the workflow-managed study lifecycle and use
  clearly named ignored result roots. Record the source revision, config,
  seeds, device assignment, intervention, primary readout, and limitations
  needed to interpret and reproduce a useful result.
- Label reduced-coverage and direct-run outputs `exploratory_noncanonical`.
  They may guide the next experiment, but they are not finalized evidence and
  must not support strong deployment or on-chip-training claims by themselves.
- Screen cheaply first and expand only promising configurations. Spend full
  assignments, endpoint repetitions, artifact verification, and formal review
  only when they can change the scientific decision or when promoting a result
  to canonical evidence.

Speed does not override the measured-data, matched-comparison, controller-port,
provenance, or honest-reporting constraints below. Preserve user data and
unrelated worktree changes.

## Local CUDA execution gate

This host (`nom-cool-2`) is expected to provide the local RTX 3090 through a
working NVIDIA driver. CUDA availability is a hard precondition for GPU work in
this tree; a CPU fallback or a skipped CUDA test is not an acceptable substitute.

- The Codex command sandbox can hide `/dev/nvidia*` from ordinary commands even
  when the host driver is healthy. An approved `nvidia-smi` invocation can run
  at host level while an ordinary Python invocation remains device-isolated, so
  `nvidia-smi` alone is not a sufficient execution check.
- Before reporting CUDA unavailable, accepting CUDA skips, or launching a GPU
  run, execute the intended Python interpreter at host level (request sandbox
  escalation) and verify an actual CUDA allocation, kernel, and synchronization:

  ```bash
  /home/filip/miniconda3/envs/py312/bin/python -c "import torch; assert torch.cuda.is_available(); x=torch.arange(4096,device='cuda',dtype=torch.float32).reshape(64,64); y=x@x.T; torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), y.device)"
  ```

- If host-level `nvidia-smi` and the escalated PyTorch canary pass, classify a
  failure from ordinary sandboxed Python as sandbox device isolation. Rerun all
  CUDA-required tests, scripts, and experiments with host-level execution; do
  not reinstall the driver, fall back to CPU, or report the driver as broken.
- If either host-level check fails, do not launch or count skipped GPU tests as
  verification. Capture the kernel/module, device-node, driver, and PyTorch
  runtime evidence and diagnose the host-level failure before proceeding.
- Verification reports for CUDA-relevant changes must state that the real
  host-level canary passed and must identify any remaining skipped tests. A
  sandbox-only `torch.cuda.is_available() == False` is never sufficient evidence
  that this host lacks working CUDA.

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
- `training/ibm_reram_program_verify.py`: explicit-RNG transcription of the
  AIHWKit 1.1.0 `SoftBoundsReferenceDevice` pulse equation, capability-limited
  controller port, one-pulse/adaptive controllers, and identity partitions.
- `training/ibm_reram_endpoint_model.py`: deployment sampler for the fitted
  target-conditioned bounded endpoint density with explicit non-corrupt
  failure, persistent corrupt-device, target-support, and clipping outcomes.
- `experiments/reram_program_verify/`: strict `characterize` composition
  root, streaming trajectory/event store, held-out endpoint fitting, Wan
  comparison, plots, and report generation.
- `experiments/reram_program_verify/local_short_launcher.py`: persistent local
  CUDA launcher and heartbeat/receipt writer for reviewed production studies.
  It invokes the public CLI for every arm; it is not an alternative numerical
  execution surface.
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
- `examples/reram_program_verify/`: operational smoke, sizing, immutable
  production, cap-128, and focused HWA-prerequisite characterization configs.
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
- Use the post-write apparent device state for the network's primary accuracy
  evaluation, model selection, and scientific conclusions; do not substitute
  accuracy computed from the hidden persistent state. At minimum, every result
  must report both apparent-state accuracy and persistent-state accuracy,
  clearly labelled and evaluated on the same examples and settings.
- The realistic P&V controller may receive only continuous targets, apparent
  normalized verify values, its own history, and calibration-partition
  population estimates. Keep persistent states and hidden device parameters
  on the logging side of the controller port.
- Treat exact persistent target reachability, verify-window intersection, and
  apparent acceptance as three different facts. Bound classification may pair
  conditioned RESET/SET states only on the analysis side; an apparent noisy
  admission must not be relabelled as a persistently reachable target.
- Preserve per-identity AIHWKit construction seeds and explicit per-trajectory
  conditioning/programming seeds. Do not replace the explicit pulse plant with
  a native CPU tile unless its cycle-to-cycle RNG becomes serializable and an
  exact-replay test is added.

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

- Exploratory runs are the default in this worktree. Before launch, record the
  question, intervention, frozen inputs and seeds, and primary readout; this
  may be a concise config or adjacent note rather than a complete study plan.
- Direct local runners and disposable canaries are allowed. Preserve failed
  attempts only when they are scientifically or operationally informative.
- State the actual coverage and limitations whenever reporting exploratory
  results; do not imply that a reduced screen is a completed study.
- Separate measured results from interpretation. Claim that on-chip training
  is needed only when the predeclared HWA-only control underperforms and a
  matched on-chip arm closes the specified gap.
- When promoting an exploratory result to workflow-managed or canonical
  evidence, follow [`docs/experiment_workflow.md`](docs/experiment_workflow.md)
  in full: predeclare the study, rerun the required coverage through the public
  CLI, use prepared `results/<study-id>/runs/<arm-id>/` roots, preserve failed
  attempts, verify artifacts, and complete scientific review and finalization.
- A concluded workflow-managed study must either be finalized into
  `docs/experimental_manifest.md` or handed off with the exact reason review or
  finalization remains pending. Use the `ebl-study-closeout` skill when it is
  available. Do not finalize exploratory runs or smoke tests as finished
  evidence.
