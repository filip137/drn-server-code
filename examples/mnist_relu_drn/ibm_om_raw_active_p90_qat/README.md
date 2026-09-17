# IBM OM raw-active p90 deployment and seven-level QAT

These configs implement the matched evaluation surface in
[`docs/ibm_om_raw_active_state_program_verify.md`](../../../docs/ibm_om_raw_active_state_program_verify.md).
They use one programmable raw active state per DRN conductance, the frozen
array-wide `g` coordinate, one baseline within each complete four-cell quad,
and the global `D90=0.10354409442884083` differential budget.

The production training arms are:

- `zero_update_quantized.json`;
- `clean_bptt_quantized_deploy.json`;
- `continuous_hwa.json`; and
- `quantized_qat.json`.

The continuous and seven-level learning rates are the independently selected
within-grid winners from
`mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1`. Both are the
upper (`x4`) boundary candidates; this limitation is recorded in that study's
selection receipt. The clean-BPTT arm retains the matched RESET-QAT clean
schedule and is selected by the same physical seven-level deployment metric.

Training HWA is deliberately mapping-only because the existing compact
endpoint fit uses the historical reference-shifted `q` coordinate and an
adaptive controller. Every checkpoint selection and held-out deployment uses
the exact boundary-conditioned one-pulse plant. The first versioned
unsupported-quad policy is an explicit structural failure represented at the
cell RESET bound during mapping-only training; it is not donor reassignment.

Held-out configs reuse assignment seed `85001` and endpoint seeds
`85101..85105`, matching the prior RESET-QAT comparison surface while changing
only the declared device coordinate, mapper, codebook, gains, and controller.
