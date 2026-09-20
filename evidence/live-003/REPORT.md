# JevRoute current-model pilot

Six frozen, unseen Python repository tasks: two easy, two medium, and two hard. Each arm used the same bounded file-edit/test tool loop; generated code ran only in a network-disabled Docker container.

| Policy | Hidden passes | Visible passes | Actual API cost | Cost per hidden pass |
|---|---:|---:|---:|---:|
| Jev three-tier B1 | 1/6 (16.7%) | 2/6 | $0.2219 | $0.2219 |
| OpenRouter Auto (same three) | 2/6 (33.3%) | 5/6 | $0.0118 | $0.0059 |
| OpenRouter Auto (unrestricted) | 2/6 (33.3%) | 4/6 | $0.0136 | $0.0068 |
| Always GLM 5.3 Flash | 2/6 (33.3%) | 3/6 | $0.0093 | $0.0046 |
| Always Gemini 3.8 Flash | 1/6 (16.7%) | 1/6 | $0.2919 | $0.2919 |
| Always GPT-5.6 Sol | 2/6 (33.3%) | 6/6 | $0.1633 | $0.0817 |
| Hindsight cheapest pass | 2/6 (33.3%) | 3/6 | $0.0093 | $0.0046 |

Total new provider-reported/accounted API cost: **$0.4902**.

## Routing diagnostics

- `easy_intervals` (easy): Jev `standard`; cheapest passing tier `economy`; fixed outcomes {'economy': True, 'standard': False, 'frontier': True}; under-routed=True; over-routed=True.
- `easy_normalize` (easy): Jev `standard`; cheapest passing tier `economy`; fixed outcomes {'economy': True, 'standard': True, 'frontier': True}; under-routed=False; over-routed=True.
- `hard_safe_paths` (hard): Jev `standard`; cheapest passing tier `None`; fixed outcomes {'economy': False, 'standard': False, 'frontier': False}; under-routed=False; over-routed=False.
- `hard_ttl_cache` (hard): Jev `frontier`; cheapest passing tier `None`; fixed outcomes {'economy': False, 'standard': False, 'frontier': False}; under-routed=False; over-routed=False.
- `medium_allocation` (medium): Jev `standard`; cheapest passing tier `None`; fixed outcomes {'economy': False, 'standard': False, 'frontier': False}; under-routed=False; over-routed=False.
- `medium_dependencies` (medium): Jev `standard`; cheapest passing tier `None`; fixed outcomes {'economy': False, 'standard': False, 'frontier': False}; under-routed=False; over-routed=False.

## Interpretation boundaries

- Current OpenRouter model metadata and actual response costs were captured at experiment start; catalog prices and promotional discounts can change.
- OpenRouter Auto uses its explicit `low` cost tier. The constrained arm can select only the three fixed candidates; the unrestricted arm can select any eligible current model.
- Jev sees the task, editable source, and visible tests, but never hidden tests or candidate outcomes. The direct B1 mapping is demand 0/economy, 1/standard, 2/frontier; missing information escalates to frontier.
- A hidden pass establishes the stated behavioral contract for these fixtures. It does not establish general maintainability, security beyond the specified path task, or performance at production scale.
- Six tasks are a feasibility screen. Report raw outcomes; do not claim statistical superiority.
- Each model gets one bounded agent session. There is no repeated-sampling estimate, and provider/model nondeterminism remains.
- Hindsight is descriptive and uses outcomes unavailable to a real router.

See `config.json`, `decisions.json`, `runs.json`, `results.json`, and per-session traces for the complete evidence.
