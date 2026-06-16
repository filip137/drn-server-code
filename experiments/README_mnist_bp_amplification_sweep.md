# MNIST BP Amplification Sweep Standard Setup

This is the standard setup for clean backpropagation MNIST DRN amplification
sweeps used for the bidirectional amplification paper. Use this configuration
for paper-comparable reruns unless there is a deliberate reason to run an
ablation.

The experiment trains DRNs with ordinary backpropagation. It does not use
EqProp, nudging updates, or memristor/write-noise injection. The amplification
factors are the only recurrence/amplification variables swept.

## Standard Command

Run the full sweep with:

```bash
python /home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py \
  --device cuda \
  --output-root /home/filip/server_code/results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc \
  --seeds 0 1 2 \
  --epochs 10 \
  --batch-size 4 \
  --num-iterations 4 \
  --input-gain 100 \
  --learning-rate 0.006 0.006 0.006 \
  --lr-decay 0.99 \
  --beta 1.0 \
  --weight-gains 1 1 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --no-download
```

Remove `--no-download` only when MNIST is not already present locally.

## Standard Hyperparameters

- Dataset: MNIST
- Input representation: signed two-channel input, shape `[2, 28, 28]`
- Architecture: DRN-XS, `[2, 28, 28] -> 100 -> 10`
- Training algorithm: BP
- Nonlinearity: perfect diode
- Input amplification factor `A`: `--input-gain 100`
- Inference iterations `T`: 4
- Training iterations `K`: 4
- Batch size: 4
- Epochs: 10
- Learning rates: `[0.006, 0.006, 0.006]`
- LR decay: 0.99
- Weight gains: `[1, 1]`
- Weight range: `[0.0, 100.0]`
- Weight init mode: `kaiming_uniform`
- Seeds: `[0, 1, 2]`
- Write noise: disabled

`--beta 1.0` is included to keep the run config aligned with the DRN-XS paper
table, but BP training does not use EqProp nudging updates.

## Preprocessing

Use the legacy recovered DRN preprocessing:

```text
Normalize(mean=0.1307, std=0.3081)
then multiply the normalized tensor by 0.3
```

In this runner that is:

```bash
--normalize-mean 0.1307 --normalize-std 0.3081 --normalize-scale 0.3
```

This is important. Using only `Normalize(mean=0.1307, std=0.3081)` without the
post-normalization `0.3` scale changes the effective input drive and was the
main mismatch found in earlier BP reruns.

## Amplification Grid

The unamplified baseline is `voltage_amp=1`, `current_amp=1`. There is no
separate no-amplification switch in this training path.

- `mnist_bp_amp_v1_c1`: `voltage_amp=1`, `current_amp=1`
- `mnist_bp_amp_v2_c1`: `voltage_amp=2`, `current_amp=1`
- `mnist_bp_amp_v4_c1`: `voltage_amp=4`, `current_amp=1`
- `mnist_bp_amp_v1_c2`: `voltage_amp=1`, `current_amp=2`
- `mnist_bp_amp_v1_c4`: `voltage_amp=1`, `current_amp=4`

The same amplification values are used for training and evaluation.

## Outputs

The standard output root is:

```text
/home/filip/server_code/results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc
```

Each run writes to:

```text
<output-root>/<run_name>/seed_<seed>/
```

Each seed directory must contain:

- `final_model.pt`
- `best_model.pt`
- `weights_final.npz`
- `weights_best.npz`
- `metrics.json`
- `accuracy_train.npy`
- `accuracy_test.npy`
- `loss_train.npy`
- `loss_test.npy`
- `config.json`

The `.npz` files save all model parameter tensors by name, including physical
nonnegative conductance tensors and biases. These files are the handoff format
for later conductance/write-noise robustness tests.

The sweep summary is:

```text
<output-root>/summary.csv
```

It contains one row per run/seed with final and best test accuracy, final
losses, best epoch, checkpoint paths, and saved weight paths.

## Parallel Runs

The standard setup is the hyperparameter set above. Parallelization should not
change the command arguments, only which run/seed each process executes.

For manual sharding, use the runner shard flags:

```bash
python /home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py \
  --device cuda \
  --output-root /home/filip/server_code/results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc \
  --seeds 0 1 2 \
  --epochs 10 \
  --batch-size 4 \
  --num-iterations 4 \
  --input-gain 100 \
  --learning-rate 0.006 0.006 0.006 \
  --lr-decay 0.99 \
  --beta 1.0 \
  --weight-gains 1 1 \
  --weight-min 0.0 \
  --weight-max 100.0 \
  --weight-init-mode kaiming_uniform \
  --normalize-mean 0.1307 \
  --normalize-std 0.3081 \
  --normalize-scale 0.3 \
  --no-download \
  --num-shards 2 \
  --shard-index 0
```

Run the same command with `--shard-index 1` for the second shard. Increase the
number of shards only if the GPU and host memory have enough headroom.

To rebuild the summary after all runs finish:

```bash
python /home/filip/server_code/experiments/train_mnist_bp_amplification_sweep.py \
  --output-root /home/filip/server_code/results/mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc \
  --seeds 0 1 2 \
  --summary-only
```
