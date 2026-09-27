# Semantic KNN router over agent-state embeddings

This experiment asks whether a nearest-neighbour router over embeddings of the visible agent state
predicts TwinRouterBench's required tier better than the metadata-only control. It reuses the
calibration split, labels, trajectory-grouped outer folds and q90 policy from the
[decomposition ablation](../twin-decomposition-001/REPORT.md). The compared variants are:

1. metadata-only logistic regression (the existing control);
2. KNN over agent-state embeddings alone;
3. KNN over embeddings plus standardized metadata features;
4. KNN over embeddings, metadata and the five compact Jev severity scalars.

Each step is serialized from router-visible fields only: the original request and the most recent
messages. Labels and row IDs are excluded. The serialized states are embedded with
`Qwen/Qwen3-Embedding-0.6B` at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, with a maximum
sequence length of 256 tokens. KNN probabilities use Dirichlet smoothing of 0.25 per class. The
number of neighbours k (3, 7 or 15), the weighting (uniform or distance) and the auxiliary feature
weight (0.25, 0.5 or 1.0) are selected only inside each outer training fold.

The primary result is five-fold nested trajectory-grouped evaluation over 189 steps from 22
development trajectories. The previously viewed 63-step holdout from 8 trajectories is secondary
diagnostic evidence only.

## Primary result

| Features | Log loss | Brier | q90 step pass | q90 trajectory pass | q90 savings | q90 combined |
|---|---:|---:|---:|---:|---:|---:|
| Metadata only (logistic control) | 1.2696 | 0.6767 | 98.41% | 90.48% | -5.92% | 58.97 |
| Embedding KNN | 1.2801 | 0.6808 | 100.00% | 100.00% | 0.00% | 62.83 |
| Embedding + metadata KNN | 1.1330 | 0.5981 | 99.47% | 96.83% | -0.63% | 62.28 |
| Embedding + metadata + compact Jev KNN | 1.1775 | 0.5977 | 99.47% | 96.83% | -0.63% | 62.28 |

Embeddings alone did not beat metadata. Log loss was slightly worse, and the q90 policy routed all
189 steps to `high`. Its 100% pass rate and higher combined score therefore come from an always-high
policy, not from routing.

Embeddings plus metadata gave the best probability quality measured so far on this split. Log loss
fell by 0.137 against the metadata control. The q90 policy still chose only `mid_high` (22 steps) or
`high` (167 steps), and failure-aware savings remained negative.

Adding the compact Jev scalars made log loss worse than embeddings plus metadata (1.1775 vs 1.1330).
The q90 decisions were identical, so Jev contributed no measurable routing value on top of the
embedding and metadata features.

No bootstrap intervals were computed for this experiment. The log-loss improvement for embeddings
plus metadata is a point estimate over 22 trajectories. It should not be treated as established
until it is confirmed on fresh trajectories.

## Argmax policy

| Features | Step pass | Trajectory pass | Savings | Under-route | Over-route |
|---|---:|---:|---:|---:|---:|
| Metadata only | 66.14% | 3.17% | -69.17% | 33.86% | 20.11% |
| Embedding KNN | 89.95% | 73.02% | -18.64% | 10.05% | 47.62% |
| Embedding + metadata KNN | 82.54% | 19.05% | -65.60% | 17.46% | 24.87% |
| Embedding + metadata + compact Jev KNN | 79.37% | 20.11% | -77.39% | 20.63% | 23.81% |

Argmax routing under-routes often enough to fail most trajectories. Under the benchmark's
failure-aware cost model, those failures cost more than the cheaper tiers save. None of the argmax
policies is usable.

## Hyperparameter selection

Every KNN variant selected k = 15 with uniform weighting in almost every outer fold. The only
exception was one fold of the Jev variant, which selected k = 7. Distance weighting always lost.
Because 15 is the largest k in the grid, the preferred setting sits on the grid boundary. The
selected models average over large neighbourhoods rather than matching individual prior states.

## Secondary reused holdout

On the already inspected holdout, embeddings plus metadata reached log loss 0.9119. Its q90 policy
preserved all 8 trajectories with 2.27% savings and a combined score of 64.46. That matches the
metadata-only and direct-Jev q90 results previously reported for this holdout. Embedding-only q90
again selected `high` for every step. The Jev variant q90 fell to 85.71% trajectory pass and -12.93%
savings. This holdout was not used for the primary conclusion.

## Decision

Do not proceed to paid dynamic SWE-bench execution. Embeddings plus metadata improve probability
quality, but no primary q90 policy achieves positive failure-aware savings. Jev adds nothing beyond
embeddings and metadata in this experiment.

If work continues, the embedding-plus-metadata representation is the strongest candidate to freeze
and validate on fresh trajectories. Extend the k grid past 15, and report trajectory-bootstrap
intervals against the metadata control.

## Cost

No API calls were made (`api_cost_usd` is 0.0). Embeddings were computed once on a Modal L4 GPU with
`scripts/embed_twin_modal.py`, and GPU cost was not recorded. The embedding cache remains under
ignored `artifacts/`. `results.json` contains the reviewable metrics, fold selections, text hashes
and holdout neighbours.
