import numpy as np


def evaluate(keys, selected, records, overhead=0.0):
    scores = [records[selected[k]][k]["score"] for k in keys]
    costs = [records[selected[k]][k]["cost"] for k in keys]
    passed = sum(scores)
    cost = sum(costs) + overhead
    avoidable = sum(records[selected[k]][k]["score"] == 0 and any(r[k]["score"] == 1 for r in records.values()) for k in keys)
    return {"n": len(keys), "passed": passed, "pass_rate": passed / len(keys),
            "replay_cost": cost, "solver_cost": sum(costs), "router_overhead": overhead,
            "cost_per_recorded_pass": cost / passed if passed else None,
            "strong_count": sum(selected[k] == "gpt-5" for k in keys),
            "avoidable_test_failures": avoidable, "quality_acceptance_rate": None,
            "scores": scores, "costs": costs}


def paired_interval(a, b, seed=20260920):
    differences = np.asarray(a) - np.asarray(b)
    samples = np.random.default_rng(seed).choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
    return {"difference": float(differences.mean()),
            "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])]}
