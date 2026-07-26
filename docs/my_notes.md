# My Notes

Updated: 2026-07-27

## Current research hypothesis

The baseline `v1/c1` networks may be more sensitive than the amplified
`v4/c1` and legacy `v4/c0.25` networks when the allowable conductance range is
limited.

The unrestricted-conductance and limited-conductance experiments should test
this directly by keeping the architecture, nonlinearity, optimizer, learning
rate, epoch budget, seeds, and checkpoint rule fixed and changing only the
conductance bounds. The relevant comparison is the accuracy loss caused by
the limited range for each amplification scheme, rather than only their
absolute final accuracies.

This is a working hypothesis, not a current conclusion.
