# Research campaigns

Use [pilots](pilots/README.md) for small, bounded tests: one note per run, without
an idea/exploration/hypothesis scaffold. Sustained research questions use the campaigns below.

| Campaign | Research question | First next step |
|---|---|---|
| [Workflow efficiency](workflow-efficiency/README.md) | Where does experiment-loop repeat work or add avoidable overhead? | Time one tiny real CPU lifecycle and review bottlenecks. |
| [Read-noise beta](read-noise-beta/README.md) | How should beta be selected across schemes for endpoint-read-noise EqProp? | Proposed exp013: refine Conv1/2/3 relative-noise and output-displacement maps to locate ours' largest layerwise alignment advantages. |
| [CIFAR-10 legacy gap](cifar10-legacy-gap/README.md) | Why is legacy behind at the current L8 endpoint, and is the gap reproducible? | Audit the comparison and distinguish coordinate/BN equivalence from optimizer history and seed variation. |
| [CIFAR-10 beta selection](cifar10-beta-selection/README.md) | How should beta be chosen for good CIFAR-10 results? | Compare EqProp/BPTT cosine across beta, then run matched 10-epoch beta comparisons. |

```text
.agents/skills/experiment-loop/SKILL.md
campaigns/pilots/<name>.md            # question through review in one note
campaigns/<research-question>/
  README.md
  ideas.md
  explorations/X-NNN-*.md
  hypotheses/H-NNN-*.md
  series/NNN-*/
    README.md
    experiments/exp-NNN-*.md
    results/*.md
  ledger.md                         # generated
```

The [experiment-loop skill](../.agents/skills/experiment-loop/SKILL.md) is the reusable
method; [artifact conventions](../.agents/skills/experiment-loop/references/artifacts.md)
provide templates and commands. [Migration plan](../docs/campaign_workflow_migration.md)
maps existing Markdown to the new structure. The lifecycle is Explorer → Codifier →
Monitor → Reviewer: the experiment record holds the scientific specification and
execution handoff and result paths, and its result note holds factual evidence followed
by review. A pilot keeps these together. The old central simulation and manifest
documents are historical references; new work does not require entries there.
run-experiment is a compatibility entry point into those same roles. Resource and
launch policy lives in [experiment_workflow](../docs/experiment_workflow.md).

Initial evidence was read from filip137/drn-server-code, branch
codex/lean-experiment-workflow, commit `6be7d41038018b225cef7d1771fdb905084b0db7` on September 24, 2026.
The linked local result bundles are not included in this setup. Historical notes are
therefore labelled imported-summary, not freshly validated-local evidence.

No training, replay, job submission or official-test evaluation is authorized or started
by installing this structure. A later explicit execution request uses the repository's
standing permission and existing scientific/operational gates.
