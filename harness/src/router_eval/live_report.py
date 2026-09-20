import json
from pathlib import Path


LABELS = {
    "always_economy": "Always GLM 5.3 Flash",
    "always_standard": "Always Gemini 3.8 Flash",
    "always_frontier": "Always GPT-5.6 Sol",
    "auto_constrained": "OpenRouter Auto (same three)",
    "auto_unrestricted": "OpenRouter Auto (unrestricted)",
    "jev_b1": "Jev three-tier B1",
    "hindsight_cheapest_pass": "Hindsight cheapest pass",
}


def write_live_report(output):
    output = Path(output)
    data = json.loads((output / "results.json").read_text(encoding="utf-8"))
    lines = ["# JevRoute current-model pilot", "",
             "Six frozen, unseen Python repository tasks: two easy, two medium, and two hard. Each arm used the same bounded file-edit/test tool loop; generated code ran only in a network-disabled Docker container.", "",
             "| Policy | Hidden passes | Visible passes | Actual API cost | Cost per hidden pass |",
             "|---|---:|---:|---:|---:|"]
    for name in ("jev_b1", "auto_constrained", "auto_unrestricted", "always_economy", "always_standard", "always_frontier", "hindsight_cheapest_pass"):
        row = data["summaries"][name]
        per_pass = "n/a" if row["cost_per_hidden_pass"] is None else f"${row['cost_per_hidden_pass']:.4f}"
        lines.append(f"| {LABELS[name]} | {row['hidden_passes']}/{row['tasks']} ({row['hidden_pass_rate']:.1%}) | {row['visible_passes']}/{row['tasks']} | ${row['total_cost']:.4f} | {per_pass} |")
    lines += ["", f"Total new provider-reported/accounted API cost: **${data['actual_api_cost']:.4f}**.", "",
              "## Routing diagnostics", ""]
    for task_id, item in data["diagnostics"].items():
        lines.append(f"- `{task_id}` ({item['difficulty']}): Jev `{item['jev_tier']}`; cheapest passing tier `{item['cheapest_passing_tier']}`; fixed outcomes {item['fixed_passes']}; under-routed={item['under_routed']}; over-routed={item['over_routed']}.")
    lines += ["", "## Interpretation boundaries", "",
              "- Current OpenRouter model metadata and actual response costs were captured at experiment start; catalog prices and promotional discounts can change.",
              "- OpenRouter Auto uses its explicit `low` cost tier. The constrained arm can select only the three fixed candidates; the unrestricted arm can select any eligible current model.",
              "- Jev sees the task, editable source, and visible tests, but never hidden tests or candidate outcomes. The direct B1 mapping is demand 0/economy, 1/standard, 2/frontier; missing information escalates to frontier.",
              "- A hidden pass establishes the stated behavioral contract for these fixtures. It does not establish general maintainability, security beyond the specified path task, or performance at production scale.",
              "- Six tasks are a feasibility screen. Report raw outcomes; do not claim statistical superiority.",
              "- Each model gets one bounded agent session. There is no repeated-sampling estimate, and provider/model nondeterminism remains.",
              "- Hindsight is descriptive and uses outcomes unavailable to a real router.", "",
              "See `config.json`, `decisions.json`, `runs.json`, `results.json`, and per-session traces for the complete evidence.", ""]
    report = output / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
