# JevRoute

Decision-only coding-task routing, with a separately packaged evaluation harness.

The caller owns the agent loop, tools, workspace, approvals and selected-model invocation. JevRoute returns a model ID and a short policy reason. It does not generate code or execute tools.

## Packages

- `jev-route`: validated input contract, Jev adapter, deterministic B0 policy, durable request journal and CLI.
- `jev-route-eval`: archived benchmark ingestion, frozen splits, published RouteLLM MF comparator, metrics and reports. Runtime does not import this package.

Local Python library/CLI is the current delivery. HTTP service, framework adapters, MCP wrappers and cloud deployment are deliberately future work. Agents can call the Python functions or CLI today. No changes to CartSavvy or its deployment are required.

## First pilot

See [the pilot report](evidence/pilot-001/REPORT.md). The first run compares live Jev classifications and published RouteLLM MF decisions against the same archived LiveCodeBench outcomes. It does not measure agent execution quality or current solver performance.

## Current-model coding-agent pilot

See [the current-model report](evidence/live-003/REPORT.md). Six frozen Python repository tasks were run through the same bounded coding-agent loop with hidden tests. The fixed tiers were GLM 5.3 Flash, Gemini 3.8 Flash, and GPT-5.6 Sol; the controls were OpenRouter Auto constrained to those models and OpenRouter Auto unrestricted, both at the documented `low` cost tier.

Jev B1 passed 1/6 hidden suites for $0.2219. Both Auto controls and always-GLM passed 2/6 for $0.0093-$0.0136. Auto selected GLM 5.3 Flash for every turn, so this pilot compares Jev with a low-cost Auto policy that behaved like an always-GLM baseline; it does not establish how other Auto cost tiers behave. Four tasks were not solved by any fixed tier, which exposes a candidate/agent-verification ceiling rather than four independent router errors. Several frontier runs passed visible tests and failed hidden edge cases, confirming that visible-test success alone is not an adequate routing label.

This six-task, one-sample pilot rejects B1 as a useful policy in its current form. It is too small to establish general model or router rankings. The next experiment should first improve task discrimination and add verification-triggered escalation, then remeasure on a fresh holdout with repeated samples.

## Run from this machine

The interpreter resolved from `D:\\projects\\Gen-AI\\loadenv.bat` is `D:\\projects\\Gen-AI\\agentic\\agentic\\Scripts\\python.exe`. No dependencies were installed into that shared environment.

PowerShell, from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\\src;$PWD\\harness\\src"
$python = 'D:\\projects\\Gen-AI\\agentic\\agentic\\Scripts\\python.exe'
# OPENROUTER_API_KEY must already be set; never put it in tracked files.
& $python scripts/restore_sources.py
& $python -m router_eval.cli run --root . --output artifacts/pilot-001
& $python -m router_eval.cli report --output artifacts/pilot-001
& $python -m pytest tests -q -p no:cacheprovider --basetemp (Join-Path $PWD ('.cache/pytest-' + [guid]::NewGuid().ToString('N')))
```

A new output directory incurs new classification/embedding calls. An existing directory reuses journaled exact requests and rejects changed experiment manifests. The USD 0.50 cap is a conservative client reservation budget, not a provider-enforced billing limit. If a request times out, reconcile the pending record before retrying. The journal is single-writer; do not run two processes against it.

Full requests, embeddings and raw model decisions remain local under ignored `artifacts/`. Journal entries omit credentials and account IDs. Prompts are sent to OpenRouter/TypeSafe or OpenRouter's embedding provider; this pilot uses public benchmark statements. Do not send private agent context without an explicit integration data policy.

## Using the router

Input JSON:

```json
{"task":{"task_id":"example-1","prompt":"Rename this local variable consistently.","language":"python"},"candidates":{"cheap":"google/gemini-2.5-flash","strong":"openai/gpt-5"}}
```

Run `python -m jev_router.cli --input task.json --journal artifacts/router.jsonl`. Output includes the selected model, policy reason and input hash. Provider errors are surfaced to the calling agent; this pilot does not silently substitute a model.

The live integration model IDs above differ from the historical archive names. The pilot names are dataset keys, not API calls to the solvers. Installed wheel entry points are `jev-route` and `jev-route-eval`.

## Agent design boundaries

Explicit small task contract; structured decisions; deterministic orchestration; caller-owned execution/context; durable restart evidence; independent evaluation; versioned policy; explicit failures. These implement relevant 12-factor-agent ideas without adding a new agent framework. Current context is prompt plus language only. Repository summaries, tools, prior failures and traces are future schema extensions, not silently discarded inputs.

## Interpreting quality

An archived pass is evidence for that benchmark's tests, not an overall quality certificate. The quality gate supports material failure and missing-evidence states but is not yet a validated semantic grader. Do not treat its aggregation unit tests as validation of a code-quality judge. Broader requirement, regression, security, instruction and hallucination metrics need executable fixtures and/or blinded review evidence.

The first pilot is an inexpensive feasibility screen. The next study should add repository tasks, sandboxed executions, seeded evaluator defects, hidden edge cases and agent traces before claiming production efficacy.

## Progressive experiments

After validating the quality evaluator, use a fresh holdout for each decision:

1. Calibrate demand thresholds from measured candidate outcomes, rather than assuming difficulty equals need for the stronger model.
2. Add bounded repository/task context and explicit missing-context handling.
3. Add uncertainty-based abstention or strong-model escalation, with measured calibration.
4. Add verification-triggered retry/escalation using actual tool/test evidence.
5. Fit a task-family-specific cost/quality policy, retaining fixed-model and OpenRouter Auto controls at explicitly recorded cost tiers.

Implement one lever at a time. Keep each pilot frozen and remeasure full task cost, quality failures, latency, and recovery costs. The first live Auto comparison is recorded in `evidence/live-003`; future experiments should include more than one Auto cost tier when budget permits.

## Sources and attribution

- [TypeSafe primitives](https://docs.typesafe.ai/introduction)
- [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) and [public dataset](https://huggingface.co/datasets/NPULH/LLMRouterBench)
- [RouteLLM](https://github.com/lm-sys/RouteLLM) and [MF checkpoint](https://huggingface.co/routellm/mf_gpt4_augmented)
- [12-factor agents](https://github.com/humanlayer/12-factor-agents)

The NumPy comparator follows RouteLLM's published MF inference equations. It is a modified implementation; PyTorch parity has not been established. See `THIRD_PARTY_NOTICES.md` and the included upstream Apache license. Checkpoint and source revisions/hashes are recorded with the pilot evidence.
