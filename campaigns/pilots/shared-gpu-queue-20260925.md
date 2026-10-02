# Shared GPU queue pilot

Question: can the shared queue admit prepared work, enforce ownership and calendar
rules, and survive a service restart without duplicate execution or model polling?

This is an operational test, with no training or scientific performance claim.
The scheduler's source lives in the reusable run-watch skill; experiment-loop
retains scientific ownership and receives the execution/validation handoff.

Decision rule: deterministic tests must cover admission, memory preference, weekday
boundaries, lost launch acknowledgements, recovery budgets and cancellation. A tiny
local/SSH GPU run must record its assigned GPU, finish, validate and survive service
restart without a second attempt. Occupied GPUs wait under the normal policy.

Cases: one local GPU smoke and one Akib SSH GPU smoke. Each allocates a small tensor
and performs bounded operations; planned peak 384 MiB plus 1 GiB headroom, expected
35 seconds, maximum 120 seconds including target validation. Controller collection
is bounded to 60 seconds. No recovery agent is needed for these test jobs.

Execution handoff: the installed
[run-watch skill](/home/filip/.codex/skills/run-watch/SKILL.md) owns both jobs.
`run-watch-queue.service` is active and enabled on nom-cool-2, with user lingering
enabled for boot startup. Prepared jobs and evidence are under
`results/shared-gpu-queue-pilot-20260925/`; remote evidence is under
`/home/filiposana/server_code/results/shared-gpu-queue-pilot-20260925/` on Akib.
Persistent queue receipts live in `~/.local/state/run-watch/queue/`.

Evidence as of September 25, 18:09 Europe/Paris:

- `queue-pilot-akib-20260925`: complete in 32.13 seconds, exactly one attempt.
  Restarting the service during computation adopted the same worker. Target
  validation and controller collection succeeded; canonical validation of
  `ssh-collected/` passed. `restart.json` records process identity across restart.
- `queue-pilot-local-20260925`: queued behind existing unrelated local training;
  no attempt or local GPU smoke result yet. Its original deadline still applies.
- All seven hosts respond; the installed service has a fresh observation without
  probe errors. No recovery model calls have occurred.
- 75 regression tests pass, including real detached CPU workers with simulated
  GPU observations. Coverage includes two Conv2 workers per 5090, rejection of a
  third, combined memory accounting, independent lease release, restart adoption,
  the weekday GPU-count cap, and Conv3's 31 GiB admission requirement. The new
  pairing/reservation policy has not yet been exercised by live 5090 training.

The Conv2/Conv3 profiles are explicit job fields. Conv3's 31 GiB reservation
includes headroom and uses scheduler accounting; it does not physically allocate
the unused memory. Existing workers retain their original exclusive locks.

Review: SSH execution, collection and restart behavior passed. The shared queue
is operational; local GPU smoke remains pending normal availability. Scientific
review stays with experiment-loop and no training conclusion follows from this pilot.
