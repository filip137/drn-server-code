# Beta diagnostics and short training

Choose beta across amplification schemes on CIFAR-10 L8, starting from blockwise EqProp/BPTT gradient
comparisons and then testing selected values in 10-epoch runs. Preserve the
calibration's explicit injected coefficient B and its summed-CE normalization.

[exp-001](experiments/exp-001-smaller-beta-cosine.md) is a read-only initialization
sweep. [exp-002](experiments/exp-002-ten-epoch-beta.md) plans actual learning; its
update rule and candidate values must be explicit before execution. Do not pool
initialization diagnostics with training outcomes or transfer MNIST beta values
without measuring their CIFAR meaning.
