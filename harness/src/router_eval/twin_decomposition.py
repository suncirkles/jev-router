"""Measure the incremental routing value of decomposed Jev task assessments."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from jev_router.classifier import EXPECTED_MODEL, MODEL
from jev_router.contracts import probabilities
from jev_router.provider import JournalClient
from router_eval.twin_calibration import (
    DIRECT_JEV_FEATURE_NAMES,
    METADATA_FEATURE_NAMES,
    _full_score,
    _latest_responses,
    _load_rows,
    _probability_matrix,
    _validate_split,
    features as direct_features,
    fit_grouped_model,
    metadata_features,
    quantile_tiers,
)
from router_eval.twinrouterbench import (
    PINNED_TWINROUTERBENCH_COMMIT,
    TIERS,
    _checkout_commit,
    _selected_cost,
    visible_state,
)


QUESTION_KEYS = (
    "scope",
    "uncertainty",
    "recovery",
    "correctness_risk",
    "verification_burden",
)

DECOMPOSED_QUESTIONS = {
    "scope": {
        "type": "choice",
        "instructions": "Assess the reasoning scope of the NEXT coding-agent call.",
        "criteria": {
            "0": "localized, routine, or a direct action with little cross-file reasoning",
            "1": "multiple files or components with meaningful interactions",
            "2": "repository-wide, architectural, or dependent on uncertain system interactions",
        },
    },
    "uncertainty": {
        "type": "choice",
        "instructions": "Assess uncertainty in what the NEXT coding-agent call should do.",
        "criteria": {
            "0": "the next action and expected result are clear from available evidence",
            "1": "some diagnosis or choice among plausible approaches remains",
            "2": "the cause, intended behavior, or viable approach remains ambiguous or conflicting",
        },
    },
    "recovery": {
        "type": "choice",
        "instructions": "Assess whether the NEXT call must recover from unsuccessful work.",
        "criteria": {
            "0": "no failed approach needs recovery",
            "1": "an attempted change, command, or test left an unresolved problem",
            "2": "repeated or consequential failures require reassessment of the approach",
        },
    },
    "correctness_risk": {
        "type": "choice",
        "instructions": "Assess the correctness risk of a plausible mistake in the NEXT call.",
        "criteria": {
            "0": "low-risk and easily reversible with direct feedback",
            "1": "edge cases, compatibility, state, or non-obvious behavior could be missed",
            "2": "subtle invariants, concurrency, security, data integrity, or broad regressions are at risk",
        },
    },
    "verification_burden": {
        "type": "choice",
        "instructions": "Assess how demanding verification is for the NEXT call.",
        "criteria": {
            "0": "one direct check gives strong evidence",
            "1": "multiple tests, edge cases, or cross-file checks are needed",
            "2": "broad regression analysis or an incomplete or indirect oracle makes verification difficult",
        },
    },
}

DECOMPOSED_FEATURE_NAMES = tuple(
    name
    for question in QUESTION_KEYS
    for name in (
        f"jev_{question}_p0",
        f"jev_{question}_p1",
        f"jev_{question}_p2",
        f"jev_{question}_entropy",
        f"jev_{question}_margin",
    )
)

TAG_NAMESPACE = "twinrouterbench-decomposed-v1"


def validate_decomposed_response(response: dict[str, Any]) -> None:
    if response.get("model") != EXPECTED_MODEL:
        raise RuntimeError("Jev resolved version changed; start a separately versioned experiment")
    try:
        answers = response["answers"]
    except (KeyError, TypeError) as exc:
        raise ValueError("unexpected Jev decomposed answer shape") from exc
    if set(answers) != set(QUESTION_KEYS):
        raise ValueError("unexpected Jev decomposed question labels")
    for key in QUESTION_KEYS:
        probabilities(answers[key]["probabilities"], ("0", "1", "2"))


def collect_response(
    client: JournalClient, row: dict[str, Any], *, reserve_per_call: float
) -> dict[str, Any]:
    payload = {"model": MODEL, "state": visible_state(row), "questions": DECOMPOSED_QUESTIONS}
    response = client.post(
        "/api/alpha/decisions",
        payload,
        f"{TAG_NAMESPACE}:{row['id']}",
        reserve=reserve_per_call,
    )
    validate_decomposed_response(response)
    return response


def decomposed_features(row: dict[str, Any], response: dict[str, Any]) -> np.ndarray:
    """Return focused Jev probabilities plus local metadata, with no target-label features."""
    validate_decomposed_response(response)
    semantic: list[float] = []
    for key in QUESTION_KEYS:
        values = response["answers"][key]["probabilities"]
        vector = [float(values[str(index)]) for index in range(3)]
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in vector)
        ordered = sorted(vector, reverse=True)
        semantic.extend([*vector, entropy, ordered[0] - ordered[1]])
    return np.concatenate([np.array(semantic, dtype=np.float64), metadata_features(row)])


def nested_oof_probabilities(
    values: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate trajectory-held-out probabilities with C selected inside each outer fold."""
    matrix = np.zeros((len(values), len(TIERS)), dtype=np.float64)
    selections: list[dict[str, Any]] = []
    outer = GroupKFold(n_splits=5)
    for fold, (train_idx, validation_idx) in enumerate(outer.split(values, labels, groups), 1):
        model, search = fit_grouped_model(
            values[train_idx], labels[train_idx], groups[train_idx]
        )
        matrix[validation_idx] = _probability_matrix(model, values[validation_idx])
        selections.append(
            {
                "fold": fold,
                "held_out_trajectories": sorted(set(groups[validation_idx].tolist())),
                "selected_C": search["selected_C"],
                "inner_candidate_log_losses": search["candidate_log_losses"],
            }
        )
    return matrix, selections


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case_pass_rate_percent",
        "case_exact_match_percent",
        "trajectory_pass_rate_percent",
        "cost_savings_score_percent",
        "combined_score_percent",
    )
    return {key: score[key] for key in keys}


def probability_summary(
    rows: list[dict[str, Any]], labels: np.ndarray, matrix: np.ndarray, section11: Any
) -> dict[str, Any]:
    argmax = np.argmax(matrix, axis=1)
    q90 = quantile_tiers(matrix, 0.90)

    def policy(predictions: np.ndarray) -> dict[str, Any]:
        high = labels == 3
        return {
            "metrics": _compact_score(_full_score(rows, predictions.tolist(), section11)),
            "selection_counts": dict(Counter(TIERS[value] for value in predictions)),
            "under_route_percent": float(100 * np.mean(predictions < labels)),
            "over_route_percent": float(100 * np.mean(predictions > labels)),
            "high_required_recall_percent": float(
                100 * np.mean(predictions[high] >= 3) if np.any(high) else 0.0
            ),
        }

    targets = np.eye(len(TIERS), dtype=np.float64)[labels]
    return {
        "log_loss": float(log_loss(labels, matrix, labels=list(range(len(TIERS))))),
        "multiclass_brier": float(np.mean(np.sum((matrix - targets) ** 2, axis=1))),
        "argmax": policy(argmax),
        "q90": policy(q90),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).resolve()
    if _checkout_commit(benchmark_root) != PINNED_TWINROUTERBENCH_COMMIT:
        raise RuntimeError("TwinRouterBench checkout differs from frozen commit")
    sys.path.insert(0, str(benchmark_root))
    section11 = importlib.import_module("main.eval.section11")

    rows = _load_rows(benchmark_root / "data" / "static" / "question_bank.jsonl")
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    _validate_split(split, rows)
    selected_ids = set(split["train"]) | set(split["holdout"])
    selected = [row for row in rows if row["instance_id"] in selected_ids]
    train_ids = set(split["train"])
    holdout_ids = set(split["holdout"])
    train = [row for row in selected if row["instance_id"] in train_ids]
    holdout = [row for row in selected if row["instance_id"] in holdout_ids]

    client = JournalClient(args.journal, cap=args.spend_cap)
    for row in selected:
        collect_response(client, row, reserve_per_call=args.reserve_per_call)
    decomposed = _latest_responses(client, TAG_NAMESPACE)
    direct = _latest_responses(JournalClient(args.direct_journal, cap=0), "twinrouterbench")
    for label, responses in (("decomposed", decomposed), ("direct", direct)):
        missing = [row["id"] for row in selected if row["id"] not in responses]
        if missing:
            raise RuntimeError(f"missing {label} responses for {len(missing)} selected rows")

    builders = {
        "metadata_only": lambda row: metadata_features(row),
        "metadata_plus_direct_jev": lambda row: direct_features(row, direct[row["id"]]),
        "metadata_plus_decomposed_jev": lambda row: decomposed_features(
            row, decomposed[row["id"]]
        ),
    }
    feature_names = {
        "metadata_only": METADATA_FEATURE_NAMES,
        "metadata_plus_direct_jev": DIRECT_JEV_FEATURE_NAMES + METADATA_FEATURE_NAMES,
        "metadata_plus_decomposed_jev": DECOMPOSED_FEATURE_NAMES + METADATA_FEATURE_NAMES,
    }
    train_y = np.array([row["target_tier_id"] for row in train], dtype=np.int64)
    holdout_y = np.array([row["target_tier_id"] for row in holdout], dtype=np.int64)
    groups = np.array([row["instance_id"] for row in train])

    variants: dict[str, Any] = {}
    for name, builder in builders.items():
        train_x = np.stack([builder(row) for row in train])
        holdout_x = np.stack([builder(row) for row in holdout])
        oof, outer_folds = nested_oof_probabilities(train_x, train_y, groups)
        final, final_search = fit_grouped_model(train_x, train_y, groups)
        holdout_probabilities = _probability_matrix(final, holdout_x)
        variants[name] = {
            "feature_names": list(feature_names[name]),
            "feature_count": len(feature_names[name]),
            "primary_nested_oof": probability_summary(train, train_y, oof, section11),
            "outer_folds": outer_folds,
            "final_training_search": final_search,
            "secondary_reused_holdout": probability_summary(
                holdout, holdout_y, holdout_probabilities, section11
            ),
        }

    result = {
        "experiment": "twinrouterbench-jev-decomposition-ablation-v1",
        "benchmark_commit": PINNED_TWINROUTERBENCH_COMMIT,
        "jev_model": EXPECTED_MODEL,
        "questions": DECOMPOSED_QUESTIONS,
        "split": split,
        "evaluation_design": {
            "primary": (
                "Five-fold outer trajectory-grouped out-of-fold evaluation on the 22 development "
                "trajectories. Each outer training fold selects logistic-regression C using its own "
                "five-fold trajectory-grouped inner cross-validation."
            ),
            "secondary": (
                "The previously inspected eight-trajectory holdout is diagnostic only and is not a "
                "fresh generalization estimate."
            ),
            "fixed_policy": "q90 selects the smallest tier covering 90% required-tier probability.",
        },
        "train_rows": len(train),
        "train_trajectories": len(train_ids),
        "holdout_rows": len(holdout),
        "holdout_trajectories": len(holdout_ids),
        "train_tier_counts": dict(Counter(TIERS[value] for value in train_y)),
        "variants": variants,
        "jev_collection": {
            "completed_calls": len(decomposed),
            "api_errors": 0,
            "selected_rows_cost_usd": _selected_cost(
                client, {row["id"] for row in selected}, TAG_NAMESPACE
            ),
            "campaign_accounted_cost_usd": client.accounted(),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--direct-journal", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spend-cap", type=float, default=0.25)
    parser.add_argument("--reserve-per-call", type=float, default=0.002)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
