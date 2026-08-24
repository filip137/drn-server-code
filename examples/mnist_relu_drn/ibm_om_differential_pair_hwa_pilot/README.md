# IBM OM differential-pair HWA pilot

These two strict configs define the internally matched canonical eight-device
pilot.  Both arms load the same tensor-identical canonicalized copy of the
frozen clean ideal-differential DRN checkpoint and use the same frozen ReLU
teacher, data/runtime seed `42`, ten continuation epochs, ideal optimizer,
fixed IBM OM assignment, pair mapper, and pulse-resolved selection deployment.

- `clean.json` performs clean FP32 continuation.
- `exact_bounds_hwa.json` exposes each minibatch to the compact IBM OM endpoint
  model on the exact same fixed differential-pair population.

The physical protocol is counterfactually repaired OM, assignment seed
`84001`, compact endpoint seed `84002`, pulse endpoint seed `84003`, adaptive
lower-from-RESET programming with a 128-pulse cap, and central-half
(`0.25` margin per side) common windows.  The mapper groups each canonical
adjacent `conductance_plus`/`conductance_minus` coordinate; no model-local
four-cell layout map is permitted.

These configs do not choose checkpoints, device-model files, or output roots.
The fail-closed launcher freezes those artifacts and supplies both
`--weights` and `--teacher-weights`.  Do not run the configs directly as
formal evidence and do not treat their results as a matched comparison with
the earlier four-device v2 study.
