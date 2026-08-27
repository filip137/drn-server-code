# IBM OM standard-level scheme screen

This strict config is consumed by
`experiments/mnist_relu_drn/ibm_om_standard_level_scheme_screen.py`.
It defines the ideal, no-write-noise screen requested after the historical
shared-RESET bounded-codebook result.

Every arm uses the same nominal spacing: adjacent active targets are exactly
four nominal OM increments apart. In the repository coordinate
`x=clip((a+1)/2,0,1)`, one increment is `dw_min/2`. The no-`r` arms anchor the
grid at each identity's sampled RESET/lower state. The fixed-`r` arms retain
the identity's sampled `r` exactly and center the active grid on it; only `a`
may move to an in-bounds integer level.

Logical sign remains a dual-rail placement decision. Active programming uses
nonnegative offsets from the selected anchor, so the four-device fixed-`r`
arm retains `[[a,r],[r,a]]` for a positive weight and swaps the roles for a
negative weight. The lower half of a centered device grid is reported as
headroom but is not counted again as additional logical sign levels.

Every logical quad uses the minimum positive-step capacity of its four cells.
A quad with no positive step is retained as a zero-only target and reported;
this first ideal screen performs no post-hoc donor reassignment. An identity
with no whole centered grid level inside its active bounds retains exact `r`
as an explicitly reported zero-only structural case instead of moving `r` or
silently inventing a partial level.

The scale pair and positive KL gain are fitted independently for each scheme
on assignment `86001` after standard-level mapping. The selected pair and
gain are then frozen on assignments `87001`, `87002`, and `87003`. Selection
uses ideal mapped calibration accuracy first and calibrated KL second. A
continuous target inside the same scheme-specific integer-level envelope is
retained only as a quantization diagnostic.

Example invocation:

```bash
python -m experiments.mnist_relu_drn.ibm_om_standard_level_scheme_screen \
  --screen-config examples/mnist_relu_drn/ibm_om_standard_level_scheme_screen/screen.json \
  --model-config examples/mnist_relu_drn/ibm_om_differential_pair_hwa_pilot/clean.json \
  --teacher-weights data/mnist_relu_teacher_fixed_init_20260816.pt \
  --output-dir results/mnist-ibm-om-standard-level-scheme-screen-20260827-v1 \
  --aihwkit-python /path/to/aihwkit/bin/python3.12 \
  --device cpu
```

This is a normalized, model-based ideal control. It is not program-and-verify,
HWA, measured-trace replay, or evidence about an absolute fabricated-device
conductance scale.
