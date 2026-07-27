# Hourly Current-Experiments Refresh

Perform one narrow maintenance pass in
`/home/filip/server_code_conv_learning_rate_protocol`.

1. Read the repository and nested `AGENTS.md` files and
   `skills/run-experiment-pipeline/SKILL.md`.
2. Validate `docs/current_experiments.md` before doing anything else.
3. Check live execution state for every marked entry using only read-only
   local, tmux, SSH, scheduler, process, and artifact-inspection commands.
   Also check for clearly identified nonterminal experiment jobs that are
   absent from the tracker.
4. Modify only `docs/current_experiments.md`, and only when verified evidence
   changes an entry's `Where` or `Status`, a reviewed result requires removal,
   or a clearly identified missing nonterminal run requires reconciliation.
5. Preserve the exact three-field contract and stable experiment markers.
   Do not change scientific intent, sweep values, evidence scope, plans,
   configs, manifests, result bundles, cards, reviews, or personal notes.
6. Use `queued` or `running` only after a current live check. Use
   `remote-closed` for terminal remote execution pending local transfer and
   validation. Remove an entry only when its required local validation,
   result card, review, and generated index entry are all verified.
7. If a host, credential, scheduler, or artifact is unavailable, do not infer
   a transition. Leave the entry unchanged and report the unavailable check
   in the final log message.
8. If nothing has changed, leave the tracker byte-identical. If it changes,
   update the top-level timestamp and the affected entry's verification text,
   then run:

   ```bash
   /home/filip/miniconda3/envs/py312/bin/python \
     skills/run-experiment-pipeline/scripts/validate_current_experiments.py \
     docs/current_experiments.md
   ```

This task is status reconciliation only. Never launch, submit, resume,
restart, cancel, kill, transfer, delete, clean up, install, commit, push, or
modify any file other than `docs/current_experiments.md`. Do not ask for
approval to broaden the task; leave unavailable or ambiguous state unchanged.
Do not spawn or delegate to subagents; perform the bounded checks directly.
