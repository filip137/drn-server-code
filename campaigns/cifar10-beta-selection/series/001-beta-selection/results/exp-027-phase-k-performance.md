---
experiment: "exp-027"
evidence: "validated-local"
summary: "V100: packing is 0.75% slower; K[4,4,3] saves17.14% update time with unchanged peak memory and passes the initial gradient screen."
verdicts: {"H-021": "contradicts", "H-022": "supports"}
---
# Packed phases versus fewer nudged iterations

All three matched batch32 profiles completed on one V100-SXM2-32GB in Slurm
458041_0, exit0:0, allocation369s. Each used35 real optimizer updates from the
same initializer; the first5 are warmups, leaving30 measured updates. Main/free,
digital and Adam operations were float32; nudged solves and gradient formula
were float64. Free T stayed[6,6,4]. No accuracy evaluation or training continuation.

| Case | Mean s/update | Time change | Allocated peak GiB | Reserved peak GiB | Projected epoch compute min |
|---|---:|---:|---:|---:|---:|
| Sequential K[6,6,4] | 2.794745 | reference | 10.728 | 13.789 | 65.54 |
| Packed +/- K[6,6,4] | 2.815721 | +0.75% | 10.854 | 15.477 | 66.03 |
| Sequential K[4,4,3] | 2.315727 | -17.14% | 10.728 | 13.789 | 54.30 |

Allocated memory counts live PyTorch allocations; reserved memory includes caching
allocator space, and neither includes all CUDA context/driver memory. Measurements
are whole-update peaks, not isolated phase state sizes. Epoch projections multiply
by1407 updates and exclude data loading, validation and beta recalibration.
They are not full-epoch measurements. Update-time standard deviations were
0.00120/0.00741/0.00146s respectively. Each successive10-update window gives the
same speed ranking, but this is one GPU allocation and one initialization.

## Numerical checks and interpretation

At the identical first minibatch (32 images), predictions, loss, boundary forces
and digital gradients matched exactly. Packing preserves the eight stored
convolution gradients: minimum cosine0.9999999999998, maximum relative L2 error
5.86e-8, norm ratios effectively1. Its fidelity criterion passes and it fits
32GB, but the hypothesized>=5% speed improvement is contradicted. Packing more
samples into the same GPU does not reduce the arithmetic count; here nudged solve
time stays near1.55s versus1.53s sequential. Fewer kernel calls did not produce a
wall-time benefit. Reserved memory increases by1.69GiB; peak allocation by0.126GiB.

Reducing K saves0.479s/update (1.207x throughput), with unchanged peak memory.
Total nudged solve time falls1.533→1.053s; free forward and autograd timing stay
unchanged. Every convolution passes the initial cosine>=0.95/norm0.9–1.1 screen:
minimum cosine0.999921, norm ratios0.987264–1.0000003, maximum relative L2 error
0.017838. This supports H-022 **only in its declared initialization/performance
regime**. The reference is original-K EqProp, not BPTT; no conclusion about later
checkpoints, accuracy or adequate settling throughout training follows.

Retain sequential phases for this V100 implementation. K[4,4,3] is the useful
candidate: approximately54min/epoch compute instead of66min. Before adopting it
for long training, check the shortened solve on trained checkpoints. No further
runs are launched by this review; the stopped five-epoch runs remain stopped.

## Evidence and validation

Root: `results/cifar-phase-k-performance-20261001-v1/`, cells`reference`, `packed`,
`reduced_k`; no failed or superseded cells. Frozen source/config/checkpoint identities
match across arms; runtime is torch2.5.0/CUDA12.2. Relative to the preceding V100
profile snapshot, only the packed estimator and profiler changed, plus test fixtures.
The manifest records both iteration vectors and phase mode. Donor coefficient
caching is enabled in all three arms, while the original float32 convolution
autograd work remains enabled. First-update gradient checks compare the float32
values actually assigned for training; CPU solver tests also compare float64
endpoints and gradients directly.

Focused frozen-source tests:24 passed. Shell/Slurm syntax and source SHA256 checks
passed before launch. All canonical bundles validate; the unchanged
`analysis/analyze.py` checks35-update coverage, precision/K/phase metadata, input
identity and matched first-update gradients. `analysis/slurm-verification.json`
confirms completed singleton task, resources and completion markers. Large source,
checkpoint, log and result artifacts remain outside Git.

[Comparison JPG](../../../../../results/cifar-phase-k-performance-20261001-v1/analysis/phase_k_performance.jpg)
· [Machine-readable summary](../../../../../results/cifar-phase-k-performance-20261001-v1/analysis/summary.json).
