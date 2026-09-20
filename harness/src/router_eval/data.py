import hashlib
import json
import random
from pathlib import Path
from jev_router.contracts import Task, digest

MODELS = ("gemini-2.5-flash", "gpt-5")


def load(cache):
    records = {}
    hashes = {}
    for model in MODELS:
        path = Path(cache) / (model + ".json")
        hashes[model] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = json.loads(path.read_text(encoding="utf-8"))["records"]
        indexed = {}
        for row in rows:
            key = digest(row["prompt"])
            if key in indexed:
                raise ValueError("Duplicate prompt in source")
            if row["score"] not in (0, 1) or row["cost"] < 0:
                raise ValueError("Pilot expects binary recorded scores and nonnegative cost")
            indexed[key] = row
        records[model] = indexed
    return records, hashes


def split(records, seed=20260920, calibration=20, evaluation=40):
    common = sorted(set.intersection(*(set(v) for v in records.values())))
    # Group duplicates by normalized original query before sampling, independent of outcomes.
    seen = set()
    eligible = []
    for key in common:
        row = records[MODELS[0]][key]
        group = " ".join(str(row["origin_query"]).split())
        if group in seen or len(row["prompt"].encode()) > 24000:
            continue
        seen.add(group)
        eligible.append(key)
    random.Random(seed).shuffle(eligible)
    if len(eligible) < calibration + evaluation:
        raise ValueError("Insufficient matched tasks")
    return {"seed": seed, "common_count": len(common), "eligible_count": len(eligible),
            "calibration": eligible[:calibration], "evaluation": eligible[calibration:calibration + evaluation]}


def task_for(records, key):
    return Task(key, records[MODELS[0]][key]["prompt"])


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Frozen manifest changed: {path.name}")
    else:
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
