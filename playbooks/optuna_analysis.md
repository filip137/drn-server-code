# Optuna Analysis SOP

## Goal
Add a standardized "Cost Decrease" analysis block to run `notes.md` files.

## Inputs
- Run directory with:
  - `time_series.pkl`
  - `notes.md`

## When to run
- Only for runs missing an `AUTO_COST_ANALYSIS_START` block in `notes.md`.

## Metrics and Definitions
Use the `Cost/test` and `Error/test` series from `time_series.pkl`.

1) Finite points
- `raw_points = len(Cost/test)`
- `finite_cost = [c for c in Cost/test if finite]`
- `finite_points = len(finite_cost)`

2) Accuracy
- `best_err = min(Error/test)` (finite only)
- `final_err = last finite Error/test`
- `best_acc = 100 - best_err`
- `final_acc = 100 - final_err`

3) Cost summary
- `initial = finite_cost[0]`
- `best = min(finite_cost)`
- `final = finite_cost[-1]`
- `best_step = first index of best (1-based)`

4) Early log-slope (first N finite points, N = min(20, finite_points))
- Fit line to `log(cost)` vs step index (1..N)
- `slope = linear_fit_slope`
- `half_life = ln(2) / -slope` if slope < 0, else "not defined"

5) Time to 20% cost drop
- First step where `cost <= 0.8 * initial`
- If none, mark "not reached"

6) Time to 90% of total improvement
- If `best < initial`, threshold = `best + 0.1 * (initial - best)`
- First step where `cost <= threshold`
- If none, mark "not reached"

7) Normalized AUC20
- `auc20 = mean(cost[:N]) / initial`

## Conclusion rule
- If `best_step == finite_points`: "Cost/test kept decreasing until the end of training."
- Else: "Cost/test improved early but then rebounded/stagnated before the end."

## Template Block
```
<!-- AUTO_COST_ANALYSIS_START -->
## Auto Analysis: Cost Decrease
Generated: YYYY-MM-DD HH:MM:SS

- Best test accuracy: XX.XX%
- Final test accuracy: XX.XX%
- Finite Cost/test points: N (raw points: M)
- Initial/Best/Final Cost(test): A / B / C
- Early log-slope (first N finite pts): S per step (more negative = faster)
- Cost half-life (from slope): H steps
- Time to 20% cost drop: step K | not reached
- Time to 90% of total cost improvement: step L | not reached
- Normalized AUC20 (lower is better): V
- Best cost reached at step B; final finite step N

### Conclusion
<conclusion text>
<!-- AUTO_COST_ANALYSIS_END -->
```

## Notes
- If `finite_points == 0`, include only accuracy, finite point count, and a conclusion:
  "Cost/test is NaN/invalid for most or all epochs; cannot assess decrease speed reliably."

