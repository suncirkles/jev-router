# JevRoute calibrated TwinRouterBench baseline

This experiment tests whether simple calibration can turn Jev's out-of-box tier probabilities into a useful coding-agent router. The split was frozen before collection: 189 steps from 22 complete SWE-bench trajectories for training and 63 steps from eight untouched trajectories for holdout evaluation. Every trajectory touched during the earlier OOB campaign was excluded.

The calibrator is balanced multinomial logistic regression over Jev's four probabilities and router-visible trajectory metadata. Regularization was selected by five-fold trajectory-grouped cross-validation. The Jev question and model version were unchanged.

## Holdout results

These are TwinRouterBench paper-v2 scores with failure-aware, cache-aware cost accounting.

| Policy | Step pass | Exact tier | Trajectory pass | Cost savings | Combined |
|---|---:|---:|---:|---:|---:|
| Jev out of box | 28.57% | 26.98% | 0.00% | -8.30% | 11.81 |
| Jev calibrated argmax | 68.25% | 58.73% | 0.00% | -65.89% | 15.27 |
| Jev calibrated q90 | 100.00% | 55.56% | 100.00% | 2.27% | 64.46 |
| Always low | 28.57% | 28.57% | 0.00% | -7.20% | 12.49 |
| Always mid | 36.51% | 7.94% | 0.00% | -6.70% | 9.44 |
| Always mid-high | 46.03% | 9.52% | 0.00% | -8.57% | 11.75 |
| Always high | 100.00% | 53.97% | 100.00% | 0.00% | 63.49 |
| Label oracle | 100.00% | 100.00% | 100.00% | 6.98% | 76.75 |

Calibration reveals signal that the direct Jev labels discard. Argmax raises step pass by 39.68 percentage points and exact-tier accuracy by 31.75 points. It still under-routes at least one critical step in every trajectory, so all trajectories fail and the failure-aware cost score becomes strongly negative.

The q90 risk policy avoids every under-route, but it selects `high` for 55 of 63 steps and `mid_high` for the other eight. It never selects `low` or `mid`. Its 2.27% savings and 0.97-point combined advantage over always-high are too small to justify dynamic execution in the current form.

For context, TwinRouterBench reports SR-KNN at 91.86% step pass, 78.76% exact tier, 84.74% trajectory pass, 56.18% savings, and 77.89 combined on all 970 static rows. That published result is a broader, in-sample upper-bound reference rather than the same holdout, but the gap is large enough to show that this Jev calibrator is not yet competitive.

## Cost and decision

The 252 Jev feature-collection calls cost **$0.06708**. There were no API errors or ambiguous requests.

Do not run the calibrated router on paid dynamic SWE-bench yet. The next lever should improve the information Jev emits: ask separate capability questions for repository scope, tool-state recovery, ambiguity, verification burden, and correctness risk, then calibrate model sufficiency from those features. A useful next static result must preserve trajectories while selecting low or mid on a material portion of safe steps.

See `split.json` for the frozen split and `results.json` for cross-validation losses, coefficients, confusion matrices, costs, and full metrics.
