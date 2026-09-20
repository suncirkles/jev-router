# JevRoute TwinRouterBench static baseline

This is the first out-of-box measurement of the Jev-based router against TwinRouterBench's April 2026 four-model pool. It uses 64 routing steps from eight complete SWE-bench trajectories selected deterministically with seed `20260920`.

| Policy | Step pass | Exact tier | Full trajectories passed | Nominal savings score |
|---|---:|---:|---:|---:|
| Jev out of box | 42.19% | 39.06% | 0/8 | 99.55% |
| Always low | 40.63% | 40.63% | 0/8 | 100.00% |
| Always mid | 46.88% | 6.25% | 1/8 | 94.65% |
| Always mid-high | 51.56% | 4.69% | 1/8 | 83.65% |
| Always high | 100.00% | 48.44% | 8/8 | 0.00% |
| Label oracle | 100.00% | 100.00% | 8/8 | 100.00% |

Jev selected `low` 49 times and `mid` 15 times. It never selected `mid_high` or `high`. Of 31 steps labeled `high`, Jev sent 20 to `low` and 11 to `mid`. Of three `mid_high` steps, it sent two to `low` and one to `mid`.

The out-of-box policy therefore does not show useful routing efficacy on this sample. Relative to always-low, it gains one passing step while losing one exact match, and neither policy passes a complete trajectory. Always-mid passes one trajectory. The core failure is systematic under-routing rather than random classification noise.

The selected 64 Jev decisions cost **$0.01709**. Total campaign accounting was **$0.02148**, including the earlier smoke trajectory and one conservatively charged request whose response became ambiguous when the local runner was interrupted. That request was not retried; its entire trajectory (`pallets__flask-5014`) was excluded from every policy in the final comparison.

## What this establishes

- The Jev adapter can consume full coding-agent trajectories and produce valid four-tier decisions without label leakage.
- The frozen Jev question plus manually stated model capability descriptions behaves approximately like an always-low policy on this sample.
- The result is a reason to calibrate the Jev-based router before paying for dynamic SWE-bench execution.

## What this does not establish

TwinRouterBench marks these SWE static records as `degradation_search_done` weak routing supervision. A step passes when the predicted tier is at least the recorded cheapest-sufficient tier. This is agreement with routing labels, not direct evidence that the selected model solves the repository task.

The benchmark's published dynamic results use a separate 100-case held-out SWE-bench execution: Claude Opus 4.6 resolves 74 cases for $54.73, SR-KNN resolves 75 for $55.61, public UncommonRoute v0.6 resolves 73 for $172.56, and trained UncommonRoute resolves 75 for $25.66. Those numbers provide the later outcome baseline; they are not directly comparable with this static eight-trajectory result.

## Next experiment

Use the remaining static trajectories as calibration data and freeze a disjoint held-out trajectory set. Keep Jev's task assessment fixed, learn a mapping from its probabilities and trajectory state to the benchmark tiers, then rerun the same controls on the holdout. A small dynamic SWE-bench run is warranted only if calibration materially improves trajectory pass rate without collapsing to always-high.

See `results.json` for the frozen prompt, source commit, confusion matrices, and exact metrics.
