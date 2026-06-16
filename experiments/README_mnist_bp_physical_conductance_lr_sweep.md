# MNIST BP Physical Conductance Amplification LR Sweep

Launch the full 60-job clean BP sweep on the `akibscomputer` tmux session:

```bash
tmux new-window -t akibscomputer -n mnist_phys_lr /home/filip/server_code/experiments/run_mnist_bp_physical_conductance_lr_sweep_akibscomputer.sh
```

The sweep trains the five amplification settings `v1/c1`, `v2/c1`, `v4/c1`, `v1/c2`, and `v1/c4` for seeds `0,1,2` and learning rates `1.25e-6`, `1.75e-6`, `2.5e-6`, and `3.5e-6`. It uses clean BP, signed MNIST inputs, 4 iterations, `input_gain=50`, batch size 4, physical conductance clipping `[0, 0.00018]`, and no write noise.

Outputs are written to:

```text
/home/filip/server_code/results/mnist_bp_amp_physical_conductance_lr_sweep/lr_<label>/<run_name>/seed_<seed>/
```

Each run directory contains `final_model.pt`, `best_model.pt`, `weights_final.npz`, `weights_best.npz`, `metrics.json`, history arrays, and `config.json`. The top-level collector writes:

```text
/home/filip/server_code/results/mnist_bp_amp_physical_conductance_lr_sweep/summary.csv
```

The collector expects 60 completed rows and verifies the required checkpoint, NPZ, metrics, history, and config artifacts before exiting successfully.
