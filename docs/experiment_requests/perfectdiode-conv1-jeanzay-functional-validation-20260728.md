# Perfect-Diode Conv1 Jean Zay Functional Validation Request

Status: approved ordinary-diagnostic process validation; no production batch

Template:
[`../experiment_request_template.md`](../experiment_request_template.md)

## User answers

### 1. Identify the work

- **Short name:** Conv1 perfect-diode Jean Zay functional validation
- **Request type:**
  - `[x]` Continue or extend an accepted operating point
  - `[ ]` Repeat an existing study
  - `[ ]` Make a changed comparison derived from an existing study
  - `[ ]` Define a new study
- **Parent experiment or result:**
  `perfectdiode-conv12-best-observed-confirmation-20260727-v1`, using the
  approved six-run Conv1 continuation as the immediate operational parent
- **Purpose:** Verify the Jean Zay Slurm, V100, module, common adapter, dataset,
  and artifact path end to end.

### 2. Say what should run

- **Cases or subset:** `pdconfirm_conv1_ours_sgd_seed0`
- **Scientific changes from the parent:** none; compute target only
- **Keep fixed from the parent:** every scientific field, including the row,
  rho values, raw parameter-specific learning rates, initialization, split,
  batch size, `T/K`, optimizer, and official-test exclusion
- **Training/evaluation budget:** validation mode with one real batch and one
  optimizer step
- **Seeds:** use the existing fixed seed 0
- **Sweep or conditional expansion:** none; exactly one Slurm array task and
  no successor production job

### 3. Say how the result should be judged

- **Decision style:**
  - `[x]` Standard continuation check: exact completion, finite numerics,
        expected artifacts, and validation trajectory; no superiority claim
  - `[ ]` Standard controlled comparison
  - `[ ]` Custom decision rule
- **Expected outcome or direction:** finite computation, a disposable
  checkpoint, a valid functional-preflight receipt, Slurm `COMPLETED/0:0`,
  and matching scheduler accounting
- **What would count as failure or negative evidence:** non-finite computation,
  missing or invalid artifacts, nonzero worker exit, or a mismatched Slurm
  resource contract
- **What would make the result inconclusive:** missing or unverifiable required
  coverage or scheduler accounting
- **Official test-set use:**
  - `[x]` Forbidden
  - `[ ]` Allowed only because the parent protocol explicitly authorizes it

### 4. Classify and place the work

- **Evidence scope:**
  - `[x]` Ordinary diagnostic
  - `[ ]` Paper-facing
  - `[ ]` Historical replay
- **Preferred compute:** Jean Zay R3 `fmu@v100`
- **Distribution preference:** one singleton `0-0` array task on one V100
- **May the executor adapt operational code and environment after a passing
  functional smoke?** yes
- **After scientific approval and passing functional checks, may the executor
  launch automatically?** yes

## Agent-resolved summary

### Catalog and parent resolution

- **Resolution:** derived from parent
- **Parent study:** `perfectdiode-conv12-best-observed-confirmation-20260727-v1`
- **Reusable cases/evidence:** `pdconfirm_conv1_ours_sgd_seed0` and its exact
  approved parent bundle
- **Reason this is a continuation, repeat, changed comparison, or new study:**
  the only change is a new compute target and bounded validation budget

### Resolved scientific study

- **Purpose/hypothesis:** the existing common Conv adapter can complete its
  one-step functional validation on Jean Zay and publish valid artifacts
- **Control:** independent completion against the fixed functional-smoke
  contract
- **Treatments/cases:** `pdconfirm_conv1_ours_sgd_seed0`
- **Scientific changes:** none
- **Inherited scientific fields:** ordinary MNIST, Conv1, perfect diode, ours
  `v4/c1`, input gain 40, batch size 16, `T=4`, `K=4`, SGD, exact parent rho
  and raw learning rates, seed 0, fixed split and initialization
- **Budget:** one real batch and optimizer step; one job
- **Decision rule:** exact finite completion with the expected checkpoint,
  functional-preflight receipt, `COMPLETED/0:0`, and matching scheduler
  accounting
- **Support/negative/inconclusive outcomes:** passing artifacts and accounting
  support target compatibility; a structured compute or contract failure is
  negative; missing evidence is inconclusive
- **Evidence scope:** ordinary process-validation diagnostic
- **Official-test policy:** forbidden
- **Conditional expansion and hard cap:** none; hard cap one job

### Scientific-gate decision

- **Prior T/K result reusable:** yes; no T/K-relevant field changed
- **Fresh T/K gate required:** no
- **Functional smoke required:** yes; this run is that target smoke

### Proposed execution

- **Targets:** Jean Zay R3
- **Job ownership:** the one selected ours/SGD case
- **Concurrency:** one
- **Functional compatibility requirements:** the reviewed module, one V100,
  readable train-only MNIST data, one finite optimizer step, checkpoint, and
  valid receipt
- **Executor operational freedom:** module/Python/path/staging adaptation only
- **Collection destination:**
  `results/.incoming/perfectdiode-conv1-jeanzay-functional-validation-20260728-v1/pdjzsmoke_conv1_ours_sgd_seed0_20260728_a`

### Resolution outcome

- **Unresolved scientific choices:** none
- **Operational warnings:** queue latency remains scheduler-controlled
- **Blocking issues:** none before live preflight
- **Resolved study status:** ready for execution under the approval below

## User review

- **Scientific summary approved:** yes
- **Execution proposal authorized:** yes
- **Corrections or constraints:** approved on 2026-07-28; no 10-epoch
  production job, only the one-step Jean Zay validation
