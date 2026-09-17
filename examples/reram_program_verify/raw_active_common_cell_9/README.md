# Raw-active p90 common-cell nine-level P&V

These configs characterize one array-wide physical-cell codebook in the
single programmable IBM OM active state.  The AIHWKit reference state is
excluded from target construction, pulse dynamics, verify values, and
inference.

The executable float32 codebook is

```text
g_k = 0.6001664102077484 + (k - 4) * 0.012380622327327728,
k = 0,...,8.
```

It is the inward-rounded form of the widest exact interval supported by 90%
of the 158,800 development cells.  Every target is programmed independently
from the same cloned, lower-bound-conditioned state with the one-pulse
controller and a 128-pulse target budget.

`development.json` uses assignment 84001, while `heldout.json` applies the
frozen codebook to assignment 85001.  `smoke.json` is operational only.

