# Experiment-loop efficiency

Question: Which parts of the Explorer → Codifier → Monitor → Reviewer lifecycle repeat work or dominate a small experiment?

Deliverable: measured phase timings and an evidence-linked, prioritized improvement list. This is an engineering workflow audit, not paper/model-performance evidence. Use existing exact_run and reporting commands; preserve all unrelated work and current scientific runs.

Scope: one two-arm CPU tiny Conv1 pilot authorized by Filip on September 25, 2026. Coordinator owns Explorer, Codifier and Reviewer; a separate Monitor exercises the existing handoff. No new control plane or production workflow changes during measurement. Targeted recovery is allowed within a ten-minute compute cap.

Decision: assess H-001 against recorded timings; record each inefficiency with observed evidence, proposed change and retained protection. One local pilot cannot characterize Slurm, SSH, failures or long-run monitoring costs. Current authority: root AGENTS.md, experiment-loop skill and docs/experiment_workflow.md including retired pre-launch gates.

Scientific question accepted: does amplification improve early learning under matched initialization, data and optimizer settings? Exp-002 tests H-002 and times the workflow for H-001. Exp-001 was superseded before execution after premature implementation.
