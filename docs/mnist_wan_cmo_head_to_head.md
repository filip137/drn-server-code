# MNIST CMO versus Wan-2022 head-to-head

## Result

This is a matched, single-device-seed comparison on all 10,000 MNIST test
examples. Both device arms use the same ordinary FP32 checkpoint, the same
two-epoch HWA checkpoint, the same DRN circuit, one-day device age, retained
physical floors, and ten epochs of full BPTT with a fresh noisy endpoint write
after every minibatch.

| Stage | CMO/HfOx | Wan-2022 | CMO minus Wan |
| --- | ---: | ---: | ---: |
| Clean ordinary FP32 DRN | 96.15% | 96.15% | 0.00 pp |
| FP32 after first device write | 96.02% | 87.80% | +8.22 pp |
| Clean HWA checkpoint, before device write | 95.67% | 95.67% | 0.00 pp |
| HWA after first device write | 96.19% | 75.78% | +20.41 pp |
| HWA + device + 10-epoch noisy full BPTT | **95.56%** | **94.23%** | **+1.33 pp** |

The BPTT row is the validation-cost-selected saved checkpoint, verified in a
fresh process. CMO selected epoch 10 and reproduced 95.56%; Wan selected
epoch 9 and reproduced 94.23%. Wan's last, non-selected epoch was 93.36%.

Relative to the first HWA-device write, ten-epoch BPTT changed CMO by
`-0.63 pp` and Wan by `+18.45 pp`. Thus, in this realization, full BPTT was
not useful for CMO but was essential for Wan. Wan nevertheless remained
`1.92 pp` below the clean FP32 baseline after recovery.

## Matched protocol

- DRN: `[1568, 100, 20]`, perfect diode, four asynchronous coordinate
  iterations, input gain 50, voltage amplification 4, and current
  amplification 0.25.
- Ordinary FP32 source: 96.15% test accuracy.
- HWA: two BPTT epochs from the FP32 source with additive Gaussian parameter
  noise, output-channel absolute-maximum scaling, `std_dev=0.03`, and seed
  101. The clean HWA checkpoint is 95.67%.
- Device seed: 17; device age: 86,400 seconds.
- Full BPTT: ten epochs, W1/W2 learning rates `0.008/0.005`, frozen hidden
  bias, and device seed 17. Gradients are evaluated at the current realized
  state; the digital shadow target is updated and both dense arrays are then
  freshly reprogrammed after every minibatch.
- Checkpoint selection: minimum validation mean cost. Under this MNIST
  configuration, `validation` resolves to the held-out test split, so the
  reported endpoint is test-selected and should not be treated as an unbiased
  generalization estimate.

The physical affine maps retain each finite floor in the DRN equations:

```text
CMO:      G_target = 9 + 79.199997 w  uS;  effective DRN G = G_realized / 88.199997
Wan-2022: G_target = 1 + 39 w         uS;  effective DRN G = G_realized / 40
```

This is one device per nonnegative DRN edge. It is not NeuRRAM's signed
two-device differential encoding. It is also an endpoint program-and-verify
experiment, not a Tiki-Taka pulse-update experiment.

## Interpretation

The finite floor is not what separates the devices here. CMO has the larger
floor ratio (`9/88.2 = 10.20%`, versus `1/40 = 2.50%` for Wan), yet it preserves
the FP32 accuracy on the first write. The affine floor adds a largely
common-mode conductance and preserves the ordering of every target before
noise; the balanced DRN representation rejects much of that common-mode term.

The stochastic endpoint error is much larger for Wan in this realization.
On the direct FP32 write, the RMS error relative to the ideal affine target is
`0.03145` DRN units for Wan and `0.00963` for CMO, a factor of 3.26. A second
independent zero-rate probe write changed CMO only from 96.02% to 95.99%, but
changed Wan from 87.80% to 90.16%. This is direct evidence that a single Wan
write has materially higher realization variance under the one-day model.

The HWA result must be interpreted cautiously. The shared HWA modifier is the
IBM-style generic additive-normal procedure, not a fit to the Wan polynomial.
It slightly lowered clean accuracy and, for Wan seed 17, moved the target to a
particularly poor first physical realization (75.78%). One realization is
insufficient to conclude that HWA generally harms Wan. The robust conclusion
from this run is narrower: the tested HWA procedure did not protect this Wan
realization, whereas noisy full BPTT recovered most of its loss.

## Provenance and exclusions

- Ordinary FP32 weights SHA-256:
  `ee7dd6c9936e051e54c38e5167133f72b64624946183b86e349b3e878ebbc27e`.
- Shared HWA weights SHA-256:
  `0055b685b1ea573dabfde127ec9a8be772bf6ce18525ab1e5db6b0d7d037afed`.
- Exact ten-epoch source commit:
  `a7bfac494b4834efc625e25ebc44f787772422cd` (clean worktree).
- Selected CMO weights SHA-256:
  `f24e0faedd342140e491aebdbec4b1330fb1112a13371cac6afdc309eb3e4ccd`.
- Selected Wan weights SHA-256:
  `bb03f4ed6dedd82b9cd48820b8fc2fcbfdcbc4b0dc2e233ebc37a03a60f37356`.
- Local raw-artifact copy: `results/mnist_wan_cmo_head_to_head_seed17_10ep/`
  (ignored by Git).

The earlier sequential 20-epoch campaign was intentionally terminated after
the user reduced the budget. Its interrupted CMO checkpoint had already
crossed into an eleventh completed epoch and is excluded from every headline
number above.

## Limitation

These are exact matched results for one device seed, not device-population
means. This is especially important for Wan, whose first two direct writes
differed by 2.36 percentage points. A population claim requires repeating the
entire physical-write and noisy-BPTT chain over matched seeds.
