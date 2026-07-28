# Perfect-Diode Conv1 Best-Rho Continuation Request

Status: approved process-validation fixture; bounded target validation pending;
not itself a tracker entry or execution receipt

Template:
[`../experiment_request_template.md`](../experiment_request_template.md)

## User answers

These answers are transcribed from Filip's 2026-07-28 request and subsequent
protocol decisions.

### 1. Identify the work

- **Short name:** Perfect-diode Conv1 six-run best-rho continuation
- **Request type:**
  - `[x]` Continue or extend an accepted operating point
  - `[ ]` Repeat an existing study
  - `[ ]` Make a changed comparison derived from an existing study
  - `[ ]` Define a new study
- **Parent experiment or result:**
  `perfectdiode-conv12-best-observed-confirmation-20260727-v1`, restricted to
  its six Conv1 entries; the learning rates originate from the accepted
  ordinary-MNIST rho selections
- **Purpose:** Run the six preselected Conv1 perfect-diode operating points
  long enough to confirm exact 10-epoch completion and obtain their
  validation trajectories.

### 2. Say what should run

- **Cases or subset:** baseline, ours, and legacy amplification, each with SGD
  and Adam; exactly the first six entries of the parent confirmation
- **Scientific changes from the parent:** none for the selected Conv1 jobs;
  select only the Conv1 subset
- **Keep fixed from the parent:** everything scientific, including raw
  learning rates, rho values, architecture, schemes, dataset, split, seeds,
  initialization, input gain, batch size, `T/K`, optimizer definitions,
  checkpoint behavior, and official-test exclusion
- **Training/evaluation budget:** 10 epochs per job
- **Seeds:** use the existing fixed seed 0
- **Sweep or conditional expansion:** none; exactly six jobs

### 3. Say how the result should be judged

- **Decision style:**
  - `[x]` Standard continuation check: exact completion, finite numerics,
        expected artifacts, and validation trajectory; no superiority claim
  - `[ ]` Standard controlled comparison
  - `[ ]` Custom decision rule
- **Expected outcome or direction:** each job completes its exact budget with
  finite outputs and valid best/final checkpoints; no cross-scheme or
  cross-optimizer ordering is expected
- **What would count as failure or negative evidence:** a job reaches a
  structured numerical, safety, or completion failure at its fixed operating
  point
- **What would make the result inconclusive:** any selected job or required
  output is missing or unverifiable
- **Official test-set use:**
  - `[x]` Forbidden
  - `[ ]` Allowed by a parent protocol

### 4. Classify and place the work

- **Evidence scope:**
  - `[x]` Ordinary diagnostic
  - `[ ]` Paper-facing
  - `[ ]` Historical replay
- **Preferred compute:** use the local GPU, Trex, and Akib
- **Distribution preference:** two jobs per host, keeping each scheme's
  SGD/Adam pair together; the executor may assign scheme pairs to hosts and
  reduce within-host concurrency when memory requires it
- **May the executor adapt operational code and environment after a passing
  functional smoke?** yes
- **After scientific approval and passing functional checks, may the executor
  launch automatically?** yes

## Agent-resolved summary

### Catalog and parent resolution

- **Resolution:** derived from parent
- **Parent study:**
  `perfectdiode-conv12-best-observed-confirmation-20260727-v1`
- **Reusable cases/evidence:** the parent's first six manifest entries and
  their source-candidate evidence
- **Reason this is a continuation:** it reuses accepted rho-selected learning
  rates and the unchanged Conv1 operating point, extends no sweep, and
  restarts each job from the already specified shared seed-0 initialization
  for the parent-defined 10-epoch confirmation

- **Catalog behavior:** the compositional catalog resolves this as a derived
  study from the exact parent, with the six-job subset classified as
  scientific selection and the three-host assignment classified as
  operational routing. It reports no unresolved scientific choice.

### Resolved scientific study

- **Purpose/hypothesis:** the six prespecified Conv1 perfect-diode
  rho-selected cells can complete 10 epochs from their shared seed-0
  initialization with finite numerics, valid checkpoints, and observable
  validation trajectories
- **Control:** each job is judged independently against its own fixed
  three-epoch source selection and exact completion contract
- **Treatments/cases, in order:**
  1. `pdconfirm_conv1_baseline_sgd_seed0`, rho `(0.009, 0.03)`
  2. `pdconfirm_conv1_baseline_adam_seed0`, rho `(0.009, 0.03)`
  3. `pdconfirm_conv1_ours_sgd_seed0`, rho `(0.009, 0.03)`
  4. `pdconfirm_conv1_ours_adam_seed0`, rho `(0.009, 0.03)`
  5. `pdconfirm_conv1_legacy_sgd_seed0`, rho `(0.003, 0.03)`
  6. `pdconfirm_conv1_legacy_adam_seed0`, rho `(0.003, 0.01)`
- **Scientific changes:** none relative to those six entries in the parent
  confirmation
- **Inherited scientific fields:**
  - ordinary MNIST with the parent's fixed training order and validation
    split;
  - Conv1, perfect diode, input gain 40, batch size 16, `T=4`, and `K=4`;
  - baseline `v1/c1`, ours `v4/c1`, and legacy `v4/c0.25`;
  - SGD and Adam definitions and exact raw parameter-specific learning rates
    from
    [`../../configs/conv/perfectdiode_conv12_best_observed_confirmation_20260727_v1.json`](../../configs/conv/perfectdiode_conv12_best_observed_confirmation_20260727_v1.json);
  - model and shuffle seed 0;
  - fresh restart from the Conv1 shared initialization, with no continuation
    from a candidate checkpoint; and
  - official-test reads forbidden
- **Budget:** 10 epochs and 34,380 optimizer steps per job; six jobs total
- **Decision rule:** require exact completion, finite loss/gradients/updates
  and states, per-epoch validation loss and accuracy, and valid
  best-validation-loss and final checkpoints; report trajectories
  descriptively with no superiority threshold
- **Support/negative/inconclusive outcomes:** support requires all six complete
  and validate; a completed structured numerical or safety failure is
  negative evidence for that fixed cell; missing or unverifiable required
  coverage is inconclusive
- **Evidence scope:** ordinary-MNIST diagnostic
- **Official-test policy:** forbidden
- **Conditional expansion and hard cap:** no expansion; hard cap six jobs

- **Evidence limitation:** five selected cells remain boundary-unresolved
  best-observed diagnostic overrides. Legacy Adam is a bracketed unpublished
  selector. These runs cannot freeze learning rates or support a cross-scheme
  superiority claim.

### Scientific-gate decision

- **Prior T/K result reusable:** yes; Conv1 perfect diode at unchanged
  baseline, ours, and legacy operating points retains the accepted `T=4`,
  `K=4` selection
- **Fresh T/K gate required:** no; this continuation changes no T/K-relevant
  scientific field
- **Functional smoke required:** yes; perform one real batch and optimizer
  step for one selected Conv1 job on each target, then verify finite numerics,
  a disposable checkpoint, and the expected output schema

### Proposed execution

- **Targets:** local GPU, Trex, and Akib
- **Job ownership:**
  - local: baseline SGD and Adam;
  - Trex: ours SGD and Adam;
  - Akib: legacy SGD and Adam
- **Concurrency:** attempt two concurrent jobs on local and Trex after their
  functional smokes; use one or two on Akib according to measured GPU memory,
  without changing job ownership
- **Functional compatibility requirements:** required Python imports,
  compatible PyTorch operations, CUDA and one visible GPU, readable ordinary
  MNIST data, writable attempt output, passing one-step smoke, finite outputs,
  and valid result/checkpoint structure
- **Executor operational freedom:** adapt Python/PyTorch/CUDA environment,
  paths, staging, tmux names, operational adapter code, and concurrency;
  record actual provenance and do not change any scientific field
- **Collection destination:** generated automatically under one local
  six-job study result, with separate local/Trex/Akib attempt provenance

### Resolution outcome

- **Unresolved scientific choices:** none identified
- **Operational warnings:** Akib's available GPU memory may require sequential
  execution of its two owned jobs
- **Blocking issues:** none in request, catalog, study validation, or dry
  execution planning; live target compatibility is intentionally decided by
  the three bounded one-batch smokes after approval
- **Resolved study status:** ready for one user review; validation dispatch
  requests are generated only after approval

## User review

- **Scientific summary approved:** yes
- **Execution proposal authorized:** yes
- **Corrections or constraints:** none

## Executor-owned details

This review fixture intentionally contains no receipt schemas, hashes, Python
paths, CUDA versions, tmux session names, or remote output paths. Those are
executor-owned operational details.
