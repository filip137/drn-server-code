# Experiment Requests

This directory contains user-facing experiment intake forms and their
agent-resolved summaries.

These files are review aids, not experiment plans, live tracker entries,
receipts, results, or launch authority. A request becomes executable only
after:

1. the agent resolves inherited and changed scientific fields;
2. the executor generates a compatible execution proposal;
3. the user approves the resolved summary and authorizes the proposal (one
   `approved` response may do both when they are presented together);
4. the executor generates the target requests and tracker state; and
5. the required functional checks pass.

Start from
[`../experiment_request_template.md`](../experiment_request_template.md).
Keep hashes, paths, environment details, receipt metadata, and process state
out of the user-answer section.
