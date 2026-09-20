# Jev task-decomposition ablation

This experiment asks whether Jev contributes routing information beyond features that can be
computed locally for free. The compared variants use the same rows, target labels, model family,
regularization candidates, outer folds, and q90 risk policy:

1. eight trajectory metadata features;
2. metadata plus the existing direct Jev tier probabilities;
3. metadata plus five focused Jev assessments: scope, uncertainty, recovery, correctness risk, and
   verification burden.

The primary result is five-fold nested trajectory-grouped evaluation over 189 steps from 22
development trajectories. Every scored trajectory is predicted by a model that did not train on it,
and regularization is selected inside each outer training fold. The previously viewed 63-step
holdout is secondary diagnostic evidence only.

## Primary result

| Features | Log loss | Brier | q90 step pass | q90 trajectory pass | q90 savings | q90 combined |
|---|---:|---:|---:|---:|---:|---:|
| Metadata only | 1.2696 | 0.6767 | 98.41% | 90.48% | -5.92% | 58.97 |
| Metadata + direct Jev | 1.2616 | 0.6777 | 98.94% | 92.06% | -5.37% | 59.64 |
| Metadata + decomposed Jev | 1.3708 | 0.7254 | 97.88% | 87.30% | -7.27% | 57.71 |

The predeclared decomposed representation did not improve routing. It made probability quality and
q90 policy performance worse than metadata alone. Direct Jev improved log loss by only 0.0079 and
q90 combined score by 0.67 points. Both q90 policies still selected high for at least 84% of steps,
and both produced negative failure-aware savings.

Cluster-bootstrap 95% intervals over the 22 trajectories include zero for every log-loss difference
versus metadata:

| Jev representation | Delta log loss vs metadata | 95% interval |
|---|---:|---:|
| Direct Jev | -0.0079 | [-0.0817, 0.0585] |
| Full decomposed Jev | +0.1013 | [-0.0290, 0.2520] |
| Reduced decomposed, post-hoc | -0.0285 | [-0.1032, 0.0457] |

Lower log loss is better. The intervals are paired, trajectory-clustered bootstrap intervals rather
than independent-row intervals.

## What the five questions measured

The focused responses varied, but most had weak association with the benchmark's required tier:

| Question | Argmax counts 0 / 1 / 2 | Correlation with required tier |
|---|---:|---:|
| Scope | 191 / 59 / 2 | 0.242 |
| Uncertainty | 114 / 138 / 0 | 0.188 |
| Recovery | 140 / 110 / 2 | 0.123 |
| Correctness risk | 111 / 122 / 19 | 0.012 |
| Verification burden | 50 / 199 / 3 | 0.100 |

Scope and uncertainty contain some ordering signal. Correctness risk, as worded, does not track the
benchmark label. This can mean the question is poorly aligned, the benchmark's weakest-sufficient
model is not well described by abstract correctness risk, or the sample is too small to distinguish
the relationship.

## Post-hoc reduced representation

The predeclared representation expanded every question into three probabilities, entropy, and margin:
25 semantic features plus eight metadata features for only 189 rows. After seeing the primary result,
one diagnostic compressed each question to its expected severity, producing five semantic scalars plus
metadata.

This post-hoc variant achieved log loss 1.2411 and q90 combined 62.28. Its q90 trajectory pass rose to
96.83% and failure-aware savings improved to -0.63%. It still selected only mid-high or high, still
lost money after failed-trajectory penalties, and its bootstrap interval includes no improvement.
These results motivate a fresh validation of a frozen low-dimensional representation; they do not
establish that it works.

## Secondary reused holdout

On the already inspected holdout, metadata-only and direct Jev q90 were identical: 100% trajectory
pass, 2.27% savings, and 64.46 combined. Full decomposed Jev q90 also preserved all trajectories but
saved only 0.56% and scored 63.63. This holdout was not used for the primary conclusion.

## Decision

Do not proceed to paid dynamic SWE-bench execution. The current evidence does not establish
incremental value from Jev beyond metadata, and no evaluated primary policy has positive
failure-aware savings.

If work continues, freeze the five-scalar representation before using new trajectories. Reduce or
replace weak questions, particularly correctness risk and verification burden, based on an explicit
hypothesis rather than repeated evaluation on these rows. A useful result must show a trajectory-held-
out improvement with positive savings, not merely a better score on the consumed holdout.

## Cost

All 252 Jev calls completed without API errors for **$0.06991**. The raw journal remains under ignored
`artifacts/twin-decomposition-001/`; `results.json` and `diagnostics.json` contain the reviewable data.
