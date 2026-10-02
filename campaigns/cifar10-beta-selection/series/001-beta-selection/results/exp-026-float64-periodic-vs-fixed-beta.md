---
experiment: "exp-026"
evidence: "validated-local"
summary: "Both arms cancelled before epoch1: approximately10.2s/update makes five epochs impractical. No validation comparison; profiling precedes further training."
verdicts: {"H-020": "inconclusive"}
---
# Partial result: training stopped for performance work

Both fresh float64 arms reached at least200 updates, but neither completed
epoch1. Approximately10.2s/update implies4hours of training per epoch before
validation. First periodic calibration took61.05s and reached all three targets;
the fixed arm had the same update cost. Calibration is not the main slowdown.

Filip requested cancellation. Queue receipts confirm stopped workers and released
reservations; charged runtime was2390.38s periodic and2394.90s fixed. All partial
bundles were collected under `results/cifar-float64-periodic-beta-20261001-v1/cells/`.
Only initialization checkpoints exist; minibatch progress cannot be resumed.
The periodic runner's failure status reflects a DataLoader worker receiving the
cancellation signal, not evidence of numerical failure. The fixed runner paused.
No validation accuracy or scientific arm comparison is available.

Training remains stopped. The [performance pilot](../../../../pilots/cifar-float64-performance-20261001.md)
removed redundant convolution gradients and repeated coefficient work. The accepted float64 implementation matches218 tensor comparisons exactly over two updates; short profiling gives10.66→7.13s/update. Training remains stopped.
