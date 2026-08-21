# Conv1 positive-only versus zero-bias diagnostic

Two matched seed-0 baseline perfect-diode Conv1 SGD arms run for ten epochs on ordinary MNIST. Both start from the same zero bias and use the accepted Conv/Dense learning rates; only `Bias_0` learning rate differs (`0.142696` versus `0`). The official test is disabled.

The predeclared practical-similarity threshold is an absolute best-validation-accuracy difference of at most `0.5` percentage points. This is a single-seed diagnostic, not paper-facing accuracy evidence.
