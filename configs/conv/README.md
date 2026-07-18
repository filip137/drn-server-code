# Canonical MNIST Conv v1 examples

These JSON files demonstrate the complete schema and deterministic sweep
grammar. They are diagnostic examples only: the operational training `T/K`,
learning-rate, batch-size, epoch, and final-inclusion protocols are still
pending. The minimal `T=1`, `K=1`, LR, epoch, batch-limit, and calibration
values exist only to make a complete diagnostic schema that can be planned;
they are not recommended settings or paper evidence. Final-category execution
is code-blocked until an exact protocol ID is approved.

Plan the example without launching training:

```bash
python -m experiments.mnist_conv sweep \
  --config configs/conv/sweep_v1.diagnostic.example.json \
  --results-root /path/to/results \
  --plan-only
```
