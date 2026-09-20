# Opaque-model identity experiment

This diagnostic tests whether Jev's routing signal depends materially on target model names. The
same 189 training steps and 63 holdout steps as `twin-calibrated-001` were reused so the comparison
is paired. Because that holdout had already been inspected, this is an explanatory experiment, not
a fresh estimate of generalization.

The harness used seed `20260922` to assign opaque identities once for the whole run:

| Opaque identity | Hidden target tier |
|---|---|
| A | mid-high |
| B | low |
| C | high |
| D | mid |

Jev saw the letter, relative cost rank, and the same capability description used in the named
experiment. It did not see a vendor name, model name, tier label, target label, or evaluator
metadata. The harness remapped Jev's choice probabilities into canonical tier order before fitting
the same balanced multinomial logistic regression. Regularization was again selected by five-fold
trajectory-grouped cross-validation, and q90 used the same predeclared risk rule.

## Paired results

These are TwinRouterBench paper-v2 failure-aware, cache-aware metrics on the same 63-step holdout.

| Policy | Presentation | Step pass | Exact tier | Trajectory pass | Cost savings | Combined |
|---|---|---:|---:|---:|---:|---:|
| Jev direct | Named | 28.57% | 26.98% | 0.00% | -8.30% | 11.81 |
| Jev direct | Opaque | 28.57% | 28.57% | 0.00% | -7.54% | 12.40 |
| Calibrated argmax | Named | 68.25% | 58.73% | 0.00% | -65.89% | 15.27 |
| Calibrated argmax | Opaque | 69.84% | 55.56% | 0.00% | -74.29% | 12.78 |
| Calibrated q90 | Named | 100.00% | 55.56% | 100.00% | 2.27% | 64.46 |
| Calibrated q90 | Opaque | 100.00% | 55.56% | 100.00% | 2.27% | 64.46 |

Across all 252 paired steps, raw named and opaque Jev outputs had the same argmax on 226 steps
(89.68%). Mean total-variation distance between their probability vectors was 0.078. Removing names
shifted average probability toward low rather than toward expensive models:

| Presentation | p(low) | p(mid) | p(mid-high) | p(high) |
|---|---:|---:|---:|---:|
| Named | 64.72% | 29.21% | 5.72% | 0.33% |
| Opaque | 70.24% | 25.37% | 4.15% | 0.23% |

Raw opaque Jev selected low on 62 of 63 holdout rows. The calibrated argmax policies differed on
7 of 63 rows. The named and opaque q90 policies made the same decision on every holdout row: 55 high
and eight mid-high. Cross-validated log loss was 1.2365 with names and 1.2497 with opaque identities,
so hiding names did not improve the signal.

## Interpretation

The target model names were not the main cause of the observed routing behavior. Jev produced highly
similar probabilities after names were removed and choices were permuted. The direct question still
treated most next calls as routine and assigned almost no probability to the strongest capability
profile.

Calibration made the conservative policy insensitive to this presentation change, but it did so by
selecting the two strongest profiles on every holdout step. This demonstrates that the harness can
route through stable opaque identities when capability and cost profiles are supplied. It does not
show useful savings, successful cold-start onboarding, or generalization to a future model.

The next information-gathering experiment should decompose the compound tier question into task
requirements such as repository scope, ambiguity, recovery state, correctness risk, and verification
burden. A metadata-only calibration control should be included to measure whether those Jev signals
add predictive value beyond trajectory position and message statistics.

## Cost and integrity

The run completed 252 of 252 Jev calls with no failures and cost **$0.06727**. The raw journal is kept
under ignored `artifacts/twin-opaque-001/`; `results.json` and `comparison.json` contain the reviewable
evidence. No target labels or evaluator metadata were included in Jev state.
