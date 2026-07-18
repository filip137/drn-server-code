# Canonical MNIST Conv v1 examples

These JSON files demonstrate the complete schema and deterministic sweep
grammar. They are diagnostic examples only. The nine hard-sigmoid input gains
and row-specific `T/K` values are frozen, while perfect-diode calibration and
`T/K` remain pending. Learning-rate, batch-size, optimizer, epoch,
final-seed, and final-inclusion protocols are also pending. The minimal `T=1`,
`K=1`, LR, epoch, batch-limit, and calibration values exist only to make a
complete diagnostic schema that can be planned; they are not recommended
settings or paper evidence. Final-category execution is code-blocked until an
exact protocol ID is approved.

Plan the example without launching training:

```bash
python -m experiments.mnist_conv sweep \
  --config configs/conv/sweep_v1.diagnostic.example.json \
  --results-root /path/to/results \
  --plan-only
```
