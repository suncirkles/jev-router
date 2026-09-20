import json
from pathlib import Path


def write_report(output):
    output = Path(output)
    data = json.loads((output / "results.json").read_text())
    rows = data["results"]
    lines = ["# JevRoute first pilot", "",
        "40 frozen evaluation tasks; 20 separate calibration tasks. Both solvers use the exact same archived prompts.", "",
        "| Policy | Recorded passes | Strong routes | Replay cost incl. router | Avoidable test failures |",
        "|---|---:|---:|---:|---:|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['passed']:g}/{r['n']} ({r['pass_rate']:.1%}) | {r['strong_count']}/{r['n']} | USD {r['replay_cost']:.4f} | {r['avoidable_test_failures']} |")
    lines += ["", f"Actual paid API charges/accounted reservations: USD {data['actual_api_cost']:.6f} (includes all 60 classifications and embeddings).",
              f"All charges provider-reported: {data['all_costs_provider_reported']}. One earlier connectivity probe cost USD 0.000013608 separately.", "",
              "Replay costs use historical per-task solver charges in the archive plus measured router overhead. Solver charges were not incurred again. Embedding overhead is allocated equally across tasks.", "",
              "## Uncertainty", "", "Paired bootstrap 95% intervals for Jev minus comparator recorded pass rate:"]
    for name, r in data["comparisons"].items():
        lines.append(f"- {name}: {r['difference']:+.1%}, interval [{r['ci95'][0]:+.1%}, {r['ci95'][1]:+.1%}].")
    lines += ["", "These intervals measure sample uncertainty, not contamination, model drift, or quality beyond tests. Forty tasks are a screening experiment, not evidence of production superiority.", "",
        "## What was and was not measured", "",
        "- Real Jev decisions through OpenRouter; resolved version fixed to typesafe/jev-1.13-20260917. Ordinal Choice is used for demand, with an argmax policy; this is disclosed instead of silently substituting an expected Score.",
        "- RouteLLM uses published mf_gpt4_augmented weights and original text-embedding-3-small embeddings. Inference is a NumPy port of the published equations, not an upstream package execution. Equation tests do not establish PyTorch numerical parity.",
        "- RouteLLM's learned GPT-4/Mixtral scoring pair is transferred to the actual Flash/GPT-5 solver pair. Default threshold and a label-free 50% calibration threshold are both shown. This is not a newly trained specialist RouteLLM.",
        "- LLMRouterBench LiveCodeBench archived Flash/GPT-5 records dated 2025-10-13. This is a single-completion replay, not an agentic repair loop or current solver run.",
        "- The dataset already contains outcomes. Splits and policy are frozen without using scores for selection/calibration; runtime receives only task ID, prompt and language. Reference scores are used only by evaluation and the explicitly hindsight oracle.",
        "- Both models may pass or fail the same task. There is no uniquely correct model label. Avoidable failures count tasks where the selected solver failed but the alternative passed.",
        "- Full quality acceptance, hallucination rate, instruction compliance and robustness remain indeterminate. Parsing and archived test scores are insufficient evidence. Eight blinded solution packets are available for a later review.",
        "- No agent execution traces are present. Tool integrity, recovery, repository changes, regressions and broader best-practice adherence cannot be inferred.",
        "- No hidden reasoning is collected or graded. Jev confidence is not treated as a calibrated probability that a solver will pass.",
        "- OpenRouter Auto was not evaluated: its current model choices cannot be fairly looked up as these two historical solver versions. A live agent study is a separate next stage.", "",
        "## Reproduction and evidence", "",
        "See split.json, config.json, thresholds.json, requests.jsonl, decisions.json, choices.json, mf_scores.json, artifact_checks.json and results.json in this directory. Source download scripts and pinned manifests are in the repository.", "",
        "Use this result to choose the next experiment, not to claim an adoption decision. The next stage should validate quality grading with seeded artifacts and then execute a small repository-task set with traces, hidden tests and material instruction checks.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return output / "REPORT.md"
