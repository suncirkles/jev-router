import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import requests

from jev_router.answers import normalize
from jev_router.classifier import EXPECTED_MODEL, MODEL, classify
from jev_router.contracts import Task
from jev_router.policy import QUESTIONS
from jev_router.provider import JournalClient
from jev_router.tiers import TierCandidates, decide_tier

from .coding_agent import run_agent
from .data import freeze
from .live_tasks import load_tasks
from .sandbox import DockerTests, IMAGE


TIERS = {
    "economy": "z-ai/glm-5.3-flash",
    "standard": "google/gemini-3.8-flash",
    "frontier": "openai/gpt-5.6-sol",
}
ARMS = {
    **{f"always_{tier}": model for tier, model in TIERS.items()},
    "auto_constrained": "openrouter/auto",
    "auto_unrestricted": "openrouter/auto",
}


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _frozen_cap(config, requested):
    frozen = float(config["spend_cap"])
    if float(requested) != frozen:
        raise RuntimeError(f"Spend cap is frozen at {frozen}; use a new output directory to change it")
    return frozen


def _cheapest_passing_tier(fixed):
    passing = [tier for tier, result in fixed.items() if result["hidden_pass"]]
    return min(passing, key=lambda tier: fixed[tier]["cost"]) if passing else None


def _model_snapshot():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    response = requests.get("https://openrouter.ai/api/v1/models",
                            headers={"Authorization": "Bearer " + key}, timeout=60)
    response.raise_for_status()
    wanted = set(TIERS.values()) | {"openrouter/auto"}
    selected = {}
    for model in response.json()["data"]:
        if model["id"] in wanted:
            selected[model["id"]] = {key: model.get(key) for key in (
                "id", "canonical_slug", "created", "pricing", "context_length",
                "supported_parameters", "expiration_date")}
    missing = wanted - set(selected)
    if missing:
        raise RuntimeError(f"Models absent from OpenRouter catalog: {sorted(missing)}")
    return selected


def _plugins(arm):
    plugin = {"id": "auto-router", "cost_tier": "low"}
    if arm == "auto_constrained":
        plugin["allowed_models"] = list(TIERS.values())
    return [plugin]


def _summary(name, task_ids, selected):
    outcomes = [selected[task_id] for task_id in task_ids]
    passed = sum(result["hidden_pass"] for result in outcomes)
    cost = sum(result["cost"] for result in outcomes)
    return {
        "name": name,
        "tasks": len(outcomes),
        "hidden_passes": passed,
        "hidden_pass_rate": passed / len(outcomes),
        "visible_passes": sum(result["visible_pass"] for result in outcomes),
        "total_cost": cost,
        "cost_per_hidden_pass": cost / passed if passed else None,
        "all_costs_measured": all(result["all_costs_measured"] for result in outcomes),
    }


def run_live(root, output, cap=2.0):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(root / "harness/live_tasks")
    code_hashes = {str(path.relative_to(root)).replace(chr(92), "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                   for folder in ("src", "harness/src") for path in (root / folder).rglob("*.py")}
    config_path = output / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {
            "experiment": "live-002-three-tier",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "docker_image": IMAGE,
            "spend_cap": cap,
            "tiers": TIERS,
            "arms": ARMS,
            "auto_cost_tier": "low",
            "jev_alias": MODEL,
            "jev_resolved": EXPECTED_MODEL,
            "questions": QUESTIONS,
            "tasks": {task.task_id: {"difficulty": task.difficulty, "fingerprint": task.fingerprint()} for task in tasks},
            "code_hashes": code_hashes,
            "model_snapshot": _model_snapshot(),
            "scope": "Six unseen Python repository tasks; one bounded tool-using session per task and arm",
        }
        freeze(config_path, config)
    expected = dict(config)
    expected["code_hashes"] = code_hashes
    expected["tasks"] = {task.task_id: {"difficulty": task.difficulty, "fingerprint": task.fingerprint()} for task in tasks}
    if expected != config:
        raise RuntimeError("Experiment code or task fixtures changed after the live run was frozen")
    client = JournalClient(output / "requests.jsonl", cap=_frozen_cap(config, cap), timeout=240)
    tests = DockerTests(config["docker_image"])
    candidates = TierCandidates(TIERS["economy"], TIERS["standard"], TIERS["frontier"])
    decisions_path = output / "decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else {}
    runs_path = output / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else {arm: {} for arm in ARMS}
    for task in tasks:
        if task.task_id not in decisions:
            route_task = Task(task.task_id, task.context())
            response = classify(route_task, client)
            decision = decide_tier(route_task, candidates, normalize(response["answers"]))
            decisions[task.task_id] = {"decision": asdict(decision), "response": response}
            _save(decisions_path, decisions)
        for arm, model in ARMS.items():
            if task.task_id in runs[arm]:
                continue
            result = run_agent(task, arm, model, output / "sessions" / arm / task.task_id,
                               client, tests, _plugins(arm) if arm.startswith("auto_") else None)
            runs[arm][task.task_id] = result
            _save(runs_path, runs)
            print(f"Completed {task.task_id} / {arm}; pass={result['hidden_pass']}; accounted USD {client.accounted():.4f}", flush=True)
    task_ids = [task.task_id for task in tasks]
    policies = {arm: {task_id: runs[arm][task_id] for task_id in task_ids} for arm in ARMS}
    jev_selected = {}
    diagnostics = {}
    tier_names = list(TIERS)
    for task_id in task_ids:
        selected_model = decisions[task_id]["decision"]["model"]
        selected_tier = next(tier for tier, model in TIERS.items() if model == selected_model)
        fixed = {tier: runs[f"always_{tier}"][task_id] for tier in tier_names}
        selected = dict(fixed[selected_tier])
        classification_cost = sum(event["cost"] for event in client.events()
                                  if event["status"] == "complete" and event["tag"] == "jev:" + task_id)
        selected["cost"] += classification_cost
        selected["router_cost"] = classification_cost
        jev_selected[task_id] = selected
        passing = [tier for tier in tier_names if fixed[tier]["hidden_pass"]]
        cheapest = _cheapest_passing_tier(fixed)
        lowest_capability = passing[0] if passing else None
        diagnostics[task_id] = {
            "difficulty": next(task.difficulty for task in tasks if task.task_id == task_id),
            "jev_tier": selected_tier,
            "cheapest_passing_tier": cheapest,
            "lowest_capability_passing_tier": lowest_capability,
            "fixed_passes": {tier: fixed[tier]["hidden_pass"] for tier in tier_names},
            "under_routed": not fixed[selected_tier]["hidden_pass"] and any(
                fixed[tier]["hidden_pass"] for tier in tier_names[tier_names.index(selected_tier) + 1:]),
            "over_routed": cheapest is not None and tier_names.index(selected_tier) > tier_names.index(cheapest),
        }
    policies["jev_b1"] = jev_selected
    oracle = {}
    for task_id in task_ids:
        passing = [runs[f"always_{tier}"][task_id] for tier in tier_names
                   if runs[f"always_{tier}"][task_id]["hidden_pass"]]
        oracle[task_id] = min(passing, key=lambda result: result["cost"]) if passing else runs["always_economy"][task_id]
    policies["hindsight_cheapest_pass"] = oracle
    summaries = {name: _summary(name, task_ids, selected) for name, selected in policies.items()}
    bundle = {
        "experiment": config["experiment"],
        "task_ids": task_ids,
        "summaries": summaries,
        "diagnostics": diagnostics,
        "auto_selected_models": {arm: {task_id: runs[arm][task_id]["selected_models"] for task_id in task_ids}
                                 for arm in ("auto_constrained", "auto_unrestricted")},
        "actual_api_cost": client.accounted(),
        "quality_scope": "Hidden behavioral tests and traces; maintainability/security outside stated requirements remain ungraded",
    }
    _save(output / "results.json", bundle)
    return bundle
