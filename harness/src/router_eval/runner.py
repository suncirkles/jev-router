import hashlib
import json
import platform
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from jev_router.answers import normalize
from jev_router.classifier import classify, MODEL, EXPECTED_MODEL
from jev_router.contracts import Candidates, digest
from jev_router.policy import QUESTIONS, decide
from jev_router.provider import JournalClient
from .data import MODELS, load, split, task_for, freeze
from .comparators import read_weights, mf_score, get_embeddings
from .quality import inspect_artifact
from .metrics import evaluate, paired_interval


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, source_hashes = load(root / ".cache")
    splits = split(records)
    freeze(output / "split.json", splits)
    manifest = json.loads((root / ".cache/routellm_manifest.json").read_text())
    code_hashes = {str(p.relative_to(root)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ("src", "harness/src") for p in (root / folder).rglob("*.py")}
    freeze(output / "config.json", {"seed": splits["seed"], "sources": source_hashes, "models": list(MODELS),
        "jev_alias": MODEL, "jev_resolved": EXPECTED_MODEL, "questions": QUESTIONS, "code_hashes": code_hashes,
        "comparator": manifest, "python": platform.python_version(), "cap": 0.50,
        "scope": "Archived LiveCodeBench test outcomes; broader quality indeterminate",
        "baseline": "choice ordinal demand argmax >= 1 OR missing >= .5; ties toward higher demand"})
    client = JournalClient(output / "requests.jsonl", cap=0.50)
    all_keys = splits["calibration"] + splits["evaluation"]
    tasks = [task_for(records, k) for k in all_keys]
    weights_path = root / ".cache/model.safetensors"
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise RuntimeError("Checkpoint checksum changed")
    weights = read_weights(weights_path)
    embeddings = get_embeddings(tasks, client)
    scores = {task.task_id: mf_score(weights, embed) for task, embed in zip(tasks, embeddings)}
    threshold = statistics.median(scores[k] for k in splits["calibration"])
    length_threshold = statistics.median(len(task_for(records, k).prompt) for k in splits["calibration"])
    freeze(output / "thresholds.json", {"mf_default": 0.5, "mf_calibrated": threshold,
        "length_calibrated": length_threshold, "objective": "50% strong allocation on calibration prompts; no outcome labels"})
    selections = {}
    decisions = {}
    candidates = Candidates(*MODELS)
    for i, task in enumerate(tasks):
        response = classify(task, client)
        decision = decide(task, candidates, normalize(response["answers"]))
        decisions[task.task_id] = {"decision": asdict(decision), "response": response}
        selections[task.task_id] = decision.model
        save(output / "decisions.json", decisions)
        print(f"Classified {i + 1}/{len(tasks)}; accounted USD {client.accounted():.6f}", flush=True)
    keys = splits["evaluation"]
    cheap, strong = MODELS
    policies = {
        "Jev B0": {k: selections[k] for k in keys},
        "RouteLLM MF default": {k: strong if scores[k] >= 0.5 else cheap for k in keys},
        "RouteLLM MF calibrated": {k: strong if scores[k] >= threshold else cheap for k in keys},
        "Always Flash": {k: cheap for k in keys},
        "Always GPT-5": {k: strong for k in keys},
        "Length control": {k: strong if len(task_for(records, k).prompt) >= length_threshold else cheap for k in keys},
        "Hindsight test oracle": {k: min(MODELS, key=lambda m: (-records[m][k]["score"], records[m][k]["cost"])) for k in keys},
    }
    completed = [e for e in client.events() if e["status"] == "complete"]
    embed_event = next(e for e in completed if e["tag"] == "routellm-embeddings")
    embed_overhead = embed_event["cost"] * len(keys) / len(tasks)
    jev_overhead = sum(e["cost"] for e in completed if e["tag"] in {"jev:" + k for k in keys})
    results = {name: evaluate(keys, selected, records,
               jev_overhead if name == "Jev B0" else embed_overhead if name.startswith("RouteLLM") else 0)
               for name, selected in policies.items()}
    comparisons = {name: paired_interval(results["Jev B0"]["scores"], result["scores"])
                   for name, result in results.items() if name not in ("Jev B0", "Hindsight test oracle")}
    save(output / "choices.json", policies)
    save(output / "mf_scores.json", scores)
    save(output / "artifact_checks.json", {k: {m: inspect_artifact(records[m][k]) for m in MODELS} for k in keys})
    # Blind packets are generated independently of outcomes. They are for review, not fabricated labels.
    packets, review_key = [], {}
    for k in keys[:4]:
        for model in MODELS:
            blind_id = digest([k, model, splits["seed"]])[:16]
            packets.append({"id": blind_id, "task_id": k, "prompt": task_for(records, k).prompt,
                            "solution": records[model][k]["prediction"]})
            review_key[blind_id] = model
    random.Random(splits["seed"]).shuffle(packets)
    save(output / "blind_review.json", packets)
    save(output / "review_key.json", review_key)
    measured = all(e["cost_measured"] for e in completed)
    bundle = {"results": results, "comparisons": comparisons,
              "actual_api_cost": client.accounted(), "all_costs_provider_reported": measured,
              "recorded_outcomes_not_rerun": True, "embedding_overhead_allocation": "batch charge divided equally per task",
              "independent_evaluation_tasks": len(keys), "calibration_tasks": len(splits["calibration"])}
    save(output / "results.json", bundle)
    return bundle
