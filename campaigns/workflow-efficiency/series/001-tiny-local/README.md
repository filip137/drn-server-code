# Tiny local workflow audit

Exp-002 follows a real scientific question through all roles: paired tiny Conv1 baseline/ours early learning. One same-path smoke per arm, then four epochs of eight two-example batches per arm, fixed sixteen-example validation prefix. Two paired hidden channels, T=K=2, Adam weight LR1e-4, frozen biases, seed0. CPU only; no paper evidence or solver-convergence claim. Exp-001 was abandoned before execution. Use existing exact_run initialization/data-order provenance, avoiding extra replay or checkpoint diagnostics.
