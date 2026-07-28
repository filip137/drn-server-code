# User Experiment Request Template

Status: user-facing intake template; not scientific approval or launch
authority

Use this form before a run. Fill only **User answers**. Plain language is
preferred. `Inherit from parent` is a complete answer when an existing study
already fixes the field.

The agent must resolve the answer into the **Agent-resolved summary**, show
every inherited value and genuine scientific change, and ask only about
remaining scientific choices. The user never fills in hashes, receipt schemas,
receipt paths, Git commits, Python paths, CUDA versions, tmux names, or output
directories.

When the agent presents the completed user answers, resolved scientific
summary, and proposed execution together, the user may reply simply
`approved`. That one response approves both review fields when nothing is
corrected; the user does not need to repeat the template or send a second
authorization.

## Compact conversational form

When the user says they want to launch an experiment without providing a
completed request, the agent must prefill this form and ask the user to
complete or correct it:

```text
Experiment:
Type: continuation | repeat | derived comparison | new
Parent:
Purpose:
Cases/subset:
Scientific changes:
Keep fixed:
Budget / seeds / sweep:
Decision style: standard continuation | standard comparison | custom
Evidence / official-test policy:
Compute / distribution:
Executor: adapt operations yes|no; launch after checks yes|no
```

The agent should replace every known field before presenting it. The user may
answer `inherit from parent`; they should never be asked to copy technical
values already fixed by the parent. If the form is fully prefilled, the agent
asks for one confirmation and includes the resolved summary and proposed
execution whenever they are available. It must not continue plan, tracker,
preflight, staging, or launch work while waiting.

## User answers

### 1. Identify the work

- **Short name:** `<your name for this work>`
- **Request type:** choose one:
  - `[ ]` Continue or extend an accepted operating point
  - `[ ]` Repeat an existing study
  - `[ ]` Make a changed comparison derived from an existing study
  - `[ ]` Define a new study
- **Parent experiment or result:** `<ID, link, description, or "new">`
- **Purpose:** `<one or two sentences describing what you want to learn or confirm>`

### 2. Say what should run

- **Cases or subset:** `<named cases, entry IDs, schemes/optimizers, or "all parent cases">`
- **Scientific changes from the parent:** `<list only changes, or "none">`
- **Keep fixed from the parent:** `<"everything else", or list exceptions>`
- **Training/evaluation budget:** `<epochs, steps, duration, or "inherit">`
- **Seeds:** `<list, "inherit", or "use the existing fixed seeds">`
- **Sweep or conditional expansion:** `<ordered values and hard cap, or "none">`

### 3. Say how the result should be judged

- **Decision style:** choose one:
  - `[ ]` Standard continuation check: exact completion, finite numerics,
        expected artifacts, and validation trajectory; no superiority claim
  - `[ ]` Standard controlled comparison: compare named treatments with the
        control using the parent metric and split
  - `[ ]` Custom decision rule
- **Expected outcome or direction:** `<plain language, or "use the selected standard default">`
- **What would count as failure or negative evidence:** `<plain language, or "use the selected standard default">`
- **What would make the result inconclusive:** `<plain language, or "missing or unverifiable required coverage">`
- **Official test-set use:** choose one:
  - `[ ]` Forbidden
  - `[ ]` Allowed only because the parent protocol explicitly authorizes it

### 4. Classify and place the work

- **Evidence scope:** choose one:
  - `[ ]` Ordinary diagnostic
  - `[ ]` Paper-facing
  - `[ ]` Historical replay
- **Preferred compute:** `<auto, local, Trex, Akib, Jean Zay, or a combination>`
- **Distribution preference:** `<auto, named cases per host, or a concurrency limit>`
- **May the executor adapt operational code and environment after a passing
  functional smoke?** `<yes/no; default yes for diagnostics>`
- **After scientific approval and passing functional checks, may the executor
  launch automatically?** `<yes/no>`

## Agent-resolved summary

The agent fills this section without inventing scientific choices.

### Catalog and parent resolution

- **Resolution:** `<exact | derived from parent | new | ambiguous>`
- **Parent study:** `<exact identity or none>`
- **Reusable cases/evidence:** `<exact identifiers>`
- **Reason this is a continuation, repeat, changed comparison, or new study:**
  `<explanation>`

### Resolved scientific study

- **Purpose/hypothesis:** `<resolved statement>`
- **Control:** `<resolved control or "independent completion per case">`
- **Treatments/cases:** `<ordered exact list>`
- **Scientific changes:** `<exact list>`
- **Inherited scientific fields:** `<dataset, model, nonlinearity, amplification, input gain, initialization, seeds, split, optimizer details, and other fixed fields>`
- **Budget:** `<exact epochs/steps and job count>`
- **Decision rule:** `<metric, split, checkpoint role, aggregation, comparison, and threshold/default>`
- **Support/negative/inconclusive outcomes:** `<resolved rules>`
- **Evidence scope:** `<resolved scope>`
- **Official-test policy:** `<resolved policy>`
- **Conditional expansion and hard cap:** `<resolved rule>`

### Scientific-gate decision

- **Prior T/K result reusable:** `<yes/no and authority>`
- **Fresh T/K gate required:** `<yes/no and exact scientific reason>`
- **Functional smoke required:** `<yes, with the smallest same-path work>`

### Proposed execution

- **Targets:** `<approved target pool>`
- **Job ownership:** `<exact cases per attempt/host>`
- **Concurrency:** `<proposed limits; may be adjusted by capability smoke>`
- **Functional compatibility requirements:** `<imports, device, data, one-step smoke, finite outputs, output schema>`
- **Executor operational freedom:** `<allowed adaptations>`
- **Collection destination:** `<agent-generated>`

### Resolution outcome

- **Unresolved scientific choices:** `<none or exact questions>`
- **Operational warnings:** `<non-blocking differences>`
- **Blocking issues:** `<none or exact blockers>`
- **Resolved study status:** `<ready for user review | needs answers>`

## User review

Complete only after reading the agent-resolved summary.

- **Scientific summary approved:** `<yes/no/pending>`
- **Execution proposal authorized:** `<yes/no/pending>`
- **Corrections or constraints:** `<none or text>`

Approval applies to the resolved scientific summary. Execution authorization
allows the executor to make only the operational adaptations listed above.
Neither authorizes silent scientific changes or an automatic retry after a
failed armed attempt.
