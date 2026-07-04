# Contaminated Old-Worktree Notes

These files preserve documentation copied from the old worktree after the WIP transfer.

They are intentionally archived here instead of merged into the canonical docs because the old worktree included simulations and interpretations affected by the `adaptive_equilibrium=True` problem.

The most useful information to salvage from these files is the practical hyperparameter trail: good learning rates, input gains, and achieved accuracies. Treat those as candidate rows to recheck or compare against. Treat solver-dependent conclusions, residual or gradient interpretations, `T/K` decisions, mechanism claims, and "best row" claims as contaminated unless independently revalidated with the corrected fixed-equilibrium protocol.

Hopfield EqProp notes are excluded from this contaminated set because they are not affected by this adaptive-equilibrium issue; the old-worktree Hopfield snapshot is stored separately under `docs/old_worktree_snapshots/`.

Source snapshot: commit `5e2d76fc` (`WIP park amplification paper work from main worktree`).
