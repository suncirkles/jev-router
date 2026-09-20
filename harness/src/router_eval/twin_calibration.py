"""Calibrate Jev routing probabilities on TwinRouterBench trajectories."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jev_router.provider import JournalClient
from router_eval.twinrouterbench import (
    JevTwinPredictor,
    IDENTITY_OPTION_TO_TIER,
    PINNED_TWINROUTERBENCH_COMMIT,
    Prediction,
    TIERS,
    _checkout_commit,
    _selected_cost,
    _static_summary,
    opaque_model_mapping,
    opaque_tier_question,
    tier_probabilities,
)


FEATURE_NAMES = (
    "jev_p_low",
    "jev_p_mid",
    "jev_p_mid_high",
    "jev_p_high",
    "jev_entropy",
    "jev_margin",
    "step_progress",
    "log_total_steps",
    "log_message_count",
    "log_character_count",
    "assistant_message_fraction",
    "tool_message_fraction",
    "has_tool_call",
    "last_role_assistant_or_tool",
)


def _load_rows(bank: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in (json.loads(line) for line in bank.read_text(encoding="utf-8").splitlines() if line)
        if row.get("benchmark") == "swebench"
    ]


def _validate_split(split: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    train, holdout, touched = (set(split[key]) for key in ("train", "holdout", "previously_touched"))
    if train & holdout or train & touched or holdout & touched:
        raise ValueError("train, holdout, and previously_touched trajectories must be disjoint")
    available = {row["instance_id"] for row in rows}
    if train | holdout | touched != available:
        raise ValueError("split must account for every SWE-bench trajectory exactly once")
    if split.get("benchmark_commit") != PINNED_TWINROUTERBENCH_COMMIT:
        raise ValueError("split benchmark commit does not match adapter pin")


def _latest_responses(client: JournalClient, tag_namespace: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in client.events():
        latest[event["tag"]] = event
    return {
        tag.removeprefix(f"{tag_namespace}:"): event["response"]
        for tag, event in latest.items()
        if tag.startswith(f"{tag_namespace}:") and event.get("status") == "complete"
    }


def _probabilities(
    response: dict[str, Any], option_to_tier: tuple[int, ...] = IDENTITY_OPTION_TO_TIER
) -> np.ndarray:
    return np.array(tier_probabilities(response, option_to_tier), dtype=np.float64)


def features(
    row: dict[str, Any],
    response: dict[str, Any],
    option_to_tier: tuple[int, ...] = IDENTITY_OPTION_TO_TIER,
) -> np.ndarray:
    """Router-visible features only; target tier and evaluator metadata are excluded."""
    probs = _probabilities(response, option_to_tier)
    messages = row["messages"]
    count = max(len(messages), 1)
    roles = [str(message.get("role", "")) for message in messages]
    serialized = json.dumps(messages, ensure_ascii=False)
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probs)
    ordered = sorted(probs, reverse=True)
    total_steps = max(int(row.get("total_steps", 1)), 1)
    return np.array(
        [
            *probs,
            entropy,
            ordered[0] - ordered[1],
            int(row.get("step_index", 1)) / total_steps,
            math.log1p(total_steps),
            math.log1p(len(messages)),
            math.log1p(len(serialized)),
            roles.count("assistant") / count,
            roles.count("tool") / count,
            float(any(message.get("tool_calls") for message in messages)),
            float(bool(roles) and roles[-1] in {"assistant", "tool"}),
        ],
        dtype=np.float64,
    )


def _probability_matrix(model: Pipeline, values: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(values)
    matrix = np.zeros((len(values), 4), dtype=np.float64)
    classes = model.named_steps["classifier"].classes_
    for source, target in enumerate(classes):
        matrix[:, int(target)] = raw[:, source]
    return matrix


def quantile_tiers(matrix: np.ndarray, quantile: float) -> np.ndarray:
    """Choose the smallest tier whose cumulative required-tier probability reaches q."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    return np.array(
        [int(np.searchsorted(np.cumsum(row), quantile, side="left")) for row in matrix],
        dtype=np.int64,
    )


def _new_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=20260921,
                ),
            ),
        ]
    )


def fit_grouped_model(values: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[Pipeline, dict[str, Any]]:
    """Select regularization by trajectory-grouped cross-validated log loss."""
    candidates = (0.01, 0.1, 1.0, 10.0)
    folds = GroupKFold(n_splits=5)
    losses: dict[str, float] = {}
    for c_value in candidates:
        fold_losses: list[float] = []
        for train_idx, validation_idx in folds.split(values, labels, groups):
            model = _new_model(c_value)
            model.fit(values[train_idx], labels[train_idx])
            predicted = _probability_matrix(model, values[validation_idx])
            fold_losses.append(log_loss(labels[validation_idx], predicted, labels=[0, 1, 2, 3]))
        losses[str(c_value)] = float(np.mean(fold_losses))
    selected = min(candidates, key=lambda value: (losses[str(value)], value))
    final = _new_model(selected)
    final.fit(values, labels)
    return final, {"candidate_log_losses": losses, "selected_C": selected, "folds": 5}


def _eval_rows(rows: list[dict[str, Any]], predictions: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "benchmark": row["benchmark"],
            "gold_tier_id": row["target_tier_id"],
            "pred_tier_id": int(prediction),
            "match": int(prediction) == row["target_tier_id"],
            "passed": int(prediction) >= row["target_tier_id"],
            "instance_id": row["instance_id"],
            "step_index": row["step_index"],
            "total_steps": row["total_steps"],
            "messages": row["messages"],
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]


def _score(rows: list[dict[str, Any]], predictions: list[int], section11: Any) -> dict[str, Any]:
    records = _eval_rows(rows, predictions)
    correct = sum(record["match"] for record in records)
    return _static_summary(records, [], correct, section11)


def _full_score(rows: list[dict[str, Any]], predictions: list[int], section11: Any) -> dict[str, Any]:
    """TwinRouterBench paper-v2 failure-aware, cache-aware score."""
    return section11.compute_v2_scores(_eval_rows(rows, predictions))


def _model_artifact(model: Pipeline) -> dict[str, Any]:
    scale = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    return {
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scale.mean_.tolist(),
        "scaler_scale": scale.scale_.tolist(),
        "classes": classifier.classes_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "C": classifier.C,
        "class_weight": classifier.class_weight,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root).resolve()
    if _checkout_commit(benchmark_root) != PINNED_TWINROUTERBENCH_COMMIT:
        raise RuntimeError("TwinRouterBench checkout differs from frozen commit")
    sys.path.insert(0, str(benchmark_root))
    section11 = importlib.import_module("main.eval.section11")

    bank = benchmark_root / "data" / "static" / "question_bank.jsonl"
    rows = _load_rows(bank)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    _validate_split(split, rows)
    selected_ids = set(split["train"]) | set(split["holdout"])
    selected = [row for row in rows if row["instance_id"] in selected_ids]

    opaque_seed = getattr(args, "opaque_seed", None)
    option_to_tier = (
        opaque_model_mapping(opaque_seed) if opaque_seed is not None else IDENTITY_OPTION_TO_TIER
    )
    question = opaque_tier_question(option_to_tier) if opaque_seed is not None else None
    tag_namespace = (
        f"twinrouterbench-opaque-{opaque_seed}" if opaque_seed is not None else "twinrouterbench"
    )
    client = JournalClient(args.journal, cap=args.spend_cap)
    predictor_kwargs: dict[str, Any] = {
        "option_to_tier": option_to_tier,
        "tag_namespace": tag_namespace,
    }
    if question is not None:
        predictor_kwargs["question"] = question
    predictor = JevTwinPredictor(client, **predictor_kwargs)
    for row in selected:
        predictor.predict(row)
    responses = _latest_responses(client, tag_namespace)
    missing = [row["id"] for row in selected if row["id"] not in responses]
    if missing:
        raise RuntimeError(f"missing completed Jev responses for {len(missing)} selected rows")

    train = [row for row in selected if row["instance_id"] in set(split["train"])]
    holdout = [row for row in selected if row["instance_id"] in set(split["holdout"])]
    train_x = np.stack(
        [features(row, responses[row["id"]], option_to_tier) for row in train]
    )
    train_y = np.array([row["target_tier_id"] for row in train], dtype=np.int64)
    train_groups = np.array([row["instance_id"] for row in train])
    holdout_x = np.stack(
        [features(row, responses[row["id"]], option_to_tier) for row in holdout]
    )

    model, cross_validation = fit_grouped_model(train_x, train_y, train_groups)
    calibrated_probabilities = _probability_matrix(model, holdout_x)
    direct = [
        int(np.argmax(_probabilities(responses[row["id"]], option_to_tier))) for row in holdout
    ]
    argmax = np.argmax(calibrated_probabilities, axis=1).tolist()
    safe = quantile_tiers(calibrated_probabilities, 0.90).tolist()
    policies: dict[str, list[int]] = {
        "jev_oob": direct,
        "jev_calibrated_argmax": argmax,
        "jev_calibrated_q90": safe,
        **{f"always_{tier}": [tier_id] * len(holdout) for tier_id, tier in enumerate(TIERS)},
        "label_oracle": [row["target_tier_id"] for row in holdout],
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": (
            "twinrouterbench-static-jev-opaque-calibrated-v1"
            if opaque_seed is not None
            else split["experiment"]
        ),
        "benchmark_commit": PINNED_TWINROUTERBENCH_COMMIT,
        "identity_blinding": {
            "enabled": opaque_seed is not None,
            "seed": opaque_seed,
            "option_to_tier": {
                chr(ord("A") + option): TIERS[tier_id]
                for option, tier_id in enumerate(option_to_tier)
            },
            "question": question,
            "tag_namespace": tag_namespace,
        },
        "split": split,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_tier_counts": dict(Counter(TIERS[value] for value in train_y)),
        "holdout_tier_counts": dict(Counter(row["target_tier"] for row in holdout)),
        "cross_validation": cross_validation,
        "model": _model_artifact(model),
        "risk_policies": {
            "argmax": "Most probable calibrated required tier.",
            "q90": "Smallest tier covering 90% of calibrated required-tier probability mass.",
        },
        "selected_rows_jev_cost_usd": _selected_cost(
            client, {row["id"] for row in selected}, tag_namespace
        ),
        "campaign_accounted_cost_usd": client.accounted(),
        "holdout": {name: _score(holdout, predictions, section11) for name, predictions in policies.items()},
    }
    if args.full_cost_scores:
        result["holdout_v2"] = {
            name: _full_score(holdout, predictions, section11)
            for name, predictions in policies.items()
        }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spend-cap", type=float, default=0.20)
    parser.add_argument("--full-cost-scores", action="store_true")
    parser.add_argument("--opaque-seed", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
