# AGENTS

## Purpose
This repository contains code and tooling for coordinate-descent simulations of DRNs with dissipative non-linearities.

## Scope and Boundaries
- In scope: Python source, configs, scripts, and documentation.
- Out of scope: large generated outputs and datasets (see `.gitignore`).

## Repo Layout (key)
- `model/`: core model components
- `training/`: training loops and monitoring
- `labs/`: experiments and utilities
- `plotting_functions/`: analysis/plotting helpers
- `playbooks/`: SOPs for repeatable tasks
- `docs/`: lightweight state and notes

## Playbooks
- Optuna analysis SOP: `playbooks/optuna_analysis.md`
- Task overrides: `playbooks/AGENTS.override.md`

## Data and Artifacts
Generated outputs live under folders like `simulation_results*`, `papers/`, and `labs/cases/` and are ignored by git.

