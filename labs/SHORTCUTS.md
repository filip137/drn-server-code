# Shortcuts

Run commands from the repository root so they always use the active worktree.

## NPZ comparison

```bash
labs/tools/compare_npz.sh run.npz spice.npz
```

Experiment execution uses `python -m ebl`; cross-worktree or multi-stage
execution uses `python -m ebl campaign run`. There are intentionally no
“latest run” or checkout-absolute experiment wrappers.
