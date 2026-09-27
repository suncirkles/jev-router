"""Trajectory-held-out semantic KNN experiment for TwinRouterBench."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from jev_router.provider import JournalClient
from router_eval.twin_calibration import (
    _full_score,
    _latest_responses,
    _load_rows,
    _validate_split,
    metadata_features,
    quantile_tiers,
)
from router_eval.twin_decomposition import QUESTION_KEYS, nested_oof_probabilities, probability_summary
from router_eval.twinrouterbench import PINNED_TWINROUTERBENCH_COMMIT, TIERS, _checkout_commit


DEFAULT_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
SERIALIZER_VERSION = "twin-agent-state-v1"
EMBEDDING_INSTRUCTION = (
    "Represent this coding-agent state for retrieval of prior states requiring similar "
    "reasoning capability and model strength."
)
K_VALUES = (3, 7, 15)
WEIGHTINGS = ("uniform", "distance")
AUX_WEIGHTS = (0.25, 0.5, 1.0)
SMOOTHING = 0.25


def _message_text(message: dict[str, Any]) -> str:
    kept = {key: message[key] for key in ("role", "content", "tool_calls") if key in message}
    return json.dumps(kept, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def serialize_state(row: dict[str, Any], *, original_limit: int = 800, recent_limit: int = 4_000) -> str:
    """Serialize router-visible state while retaining the task and the latest agent evidence."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row messages must be a list")
    user = next((message for message in messages if message.get("role") == "user"), {})
    original = str(user.get("content", ""))[:original_limit]
    recent: list[str] = []
    used = 0
    for message in reversed(messages):
        rendered = _message_text(message)
        if used + len(rendered) > recent_limit:
            remaining = recent_limit - used
            if remaining > 0:
                recent.append(rendered[-remaining:])
            break
        recent.append(rendered)
        used += len(rendered)
    return "\n".join(
        [
            f"serializer={SERIALIZER_VERSION}",
            f"benchmark={row.get('benchmark', '')}",
            f"scenario={row.get('scenario', '')}",
            f"progress={row.get('step_index', 1)}/{row.get('total_steps', 1)}",
            "original_task:",
            original,
            "recent_state_newest_first:",
            *recent,
        ]
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_or_create_embeddings(
    rows: list[dict[str, Any]],
    cache_path: Path,
    *,
    model_name: str = DEFAULT_EMBEDDER,
    revision: str = "main",
    max_seq_length: int = 256,
    batch_size: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    texts = [serialize_state(row) for row in rows]
    ids = [str(row["id"]) for row in rows]
    hashes = [_hash_text(text) for text in texts]
    metadata_path = cache_path.with_suffix(".json")
    expected = {
        "model_name": model_name,
        "requested_revision": revision,
        "serializer_version": SERIALIZER_VERSION,
        "max_seq_length": max_seq_length,
        "ids": ids,
        "text_sha256": hashes,
    }
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            with np.load(cache_path) as cached:
                return np.asarray(cached["embeddings"], dtype=np.float64), metadata

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    model.max_seq_length = max_seq_length
    embeddings = model.encode(
        texts,
        prompt=EMBEDDING_INSTRUCTION + "\nQuery: ",
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    first = model._first_module()  # sentence-transformers exposes the backing transformer here.
    resolved_revision = getattr(getattr(first, "auto_model", None), "config", None)
    resolved_revision = getattr(resolved_revision, "_commit_hash", None)
    metadata = {
        **expected,
        "resolved_revision": resolved_revision,
        "embedding_dimension": int(embeddings.shape[1]),
        "instruction": EMBEDDING_INSTRUCTION,
        "cache_format": "numpy-npz-v1",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=np.asarray(embeddings, dtype=np.float32))
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return np.asarray(embeddings, dtype=np.float64), metadata


def compact_jev_features(response: dict[str, Any]) -> np.ndarray:
    """Five expected severities from the frozen decomposed Jev questions."""
    answers = response.get("answers", {})
    if set(answers) != set(QUESTION_KEYS):
        raise ValueError("unexpected decomposed Jev answers")
    values = []
    for key in QUESTION_KEYS:
        probs = answers[key]["probabilities"]
        values.append(sum(index * float(probs[str(index)]) for index in range(3)) / 2.0)
    return np.asarray(values, dtype=np.float64)


def _combine_features(
    train_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    train_aux: np.ndarray | None,
    query_aux: np.ndarray | None,
    aux_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if train_aux is None:
        return train_embeddings, query_embeddings
    mean = train_aux.mean(axis=0)
    scale = train_aux.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_scaled = (train_aux - mean) / scale
    query_scaled = (query_aux - mean) / scale
    train = np.concatenate([train_embeddings, aux_weight * train_scaled], axis=1)
    query = np.concatenate([query_embeddings, aux_weight * query_scaled], axis=1)
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    return train, query


def knn_probabilities(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    query_values: np.ndarray,
    *,
    k: int,
    weighting: str,
    smoothing: float = SMOOTHING,
) -> np.ndarray:
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting: {weighting}")
    if k <= 0 or len(train_values) == 0:
        raise ValueError("k and training set must be non-empty")
    count = min(k, len(train_values))
    distances = np.maximum(1.0 - query_values @ train_values.T, 0.0)
    neighbors = np.argpartition(distances, count - 1, axis=1)[:, :count]
    result = np.empty((len(query_values), len(TIERS)), dtype=np.float64)
    for row_index, indices in enumerate(neighbors):
        local_distances = distances[row_index, indices]
        weights = (
            np.ones(len(indices), dtype=np.float64)
            if weighting == "uniform"
            else 1.0 / np.maximum(local_distances, 1e-6)
        )
        totals = np.full(len(TIERS), smoothing, dtype=np.float64)
        for label, weight in zip(train_labels[indices], weights, strict=True):
            totals[int(label)] += float(weight)
        result[row_index] = totals / totals.sum()
    return result


def _candidate_grid(has_aux: bool) -> list[dict[str, Any]]:
    aux_weights = AUX_WEIGHTS if has_aux else (0.0,)
    return [
        {"k": k, "weighting": weighting, "aux_weight": aux_weight}
        for k in K_VALUES
        for weighting in WEIGHTINGS
        for aux_weight in aux_weights
    ]


def select_knn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    aux: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    folds = GroupKFold(n_splits=5)
    losses: dict[str, float] = {}
    candidates = _candidate_grid(aux is not None)
    for candidate in candidates:
        fold_losses = []
        for train_idx, validation_idx in folds.split(embeddings, labels, groups):
            train, validation = _combine_features(
                embeddings[train_idx],
                embeddings[validation_idx],
                None if aux is None else aux[train_idx],
                None if aux is None else aux[validation_idx],
                candidate["aux_weight"],
            )
            predicted = knn_probabilities(
                train,
                labels[train_idx],
                validation,
                k=candidate["k"],
                weighting=candidate["weighting"],
            )
            fold_losses.append(log_loss(labels[validation_idx], predicted, labels=list(range(4))))
        key = json.dumps(candidate, sort_keys=True)
        losses[key] = float(np.mean(fold_losses))
    selected = min(candidates, key=lambda item: (losses[json.dumps(item, sort_keys=True)], json.dumps(item, sort_keys=True)))
    return selected, losses


def nested_knn_probabilities(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    aux: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = np.zeros((len(embeddings), len(TIERS)), dtype=np.float64)
    selections = []
    outer = GroupKFold(n_splits=5)
    for fold, (train_idx, validation_idx) in enumerate(outer.split(embeddings, labels, groups), 1):
        selected, losses = select_knn(
            embeddings[train_idx], labels[train_idx], groups[train_idx], None if aux is None else aux[train_idx]
        )
        train, validation = _combine_features(
            embeddings[train_idx],
            embeddings[validation_idx],
            None if aux is None else aux[train_idx],
            None if aux is None else aux[validation_idx],
            selected["aux_weight"],
        )
        matrix[validation_idx] = knn_probabilities(
            train, labels[train_idx], validation, k=selected["k"], weighting=selected["weighting"]
        )
        selections.append(
            {
                "fold": fold,
                "held_out_trajectories": sorted(set(groups[validation_idx].tolist())),
                "selected": selected,
                "candidate_log_losses": losses,
            }
        )
    return matrix, selections


def fit_predict_knn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    query_embeddings: np.ndarray,
    train_aux: np.ndarray | None = None,
    query_aux: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    selected, losses = select_knn(train_embeddings, train_labels, train_groups, train_aux)
    train, query = _combine_features(
        train_embeddings, query_embeddings, train_aux, query_aux, selected["aux_weight"]
    )
    probabilities = knn_probabilities(
        train, train_labels, query, k=selected["k"], weighting=selected["weighting"]
    )
    return probabilities, {"selected": selected, "candidate_log_losses": losses}, train, query


def neighbor_evidence(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    queries: list[dict[str, Any]],
    train_values: np.ndarray,
    query_values: np.ndarray,
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    distances = np.maximum(1.0 - query_values @ train_values.T, 0.0)
    result = []
    for query_index, query in enumerate(queries):
        nearest = np.argsort(distances[query_index])[:count]
        result.append(
            {
                "id": query["id"],
                "neighbors": [
                    {
                        "id": rows[index]["id"],
                        "instance_id": rows[index]["instance_id"],
                        "tier": TIERS[int(labels[index])],
                        "cosine_distance": float(distances[query_index, index]),
                    }
                    for index in nearest
                ],
            }
        )
    return result


def risk_sweep(
    rows: list[dict[str, Any]], labels: np.ndarray, matrix: np.ndarray, section11: Any
) -> dict[str, Any]:
    """Post-hoc operating frontier; q90 remains the fixed primary policy."""
    result = {}
    for quantile in (0.70, 0.80, 0.85, 0.90, 0.95):
        predictions = quantile_tiers(matrix, quantile)
        score = _full_score(rows, predictions.tolist(), section11)
        result[f"q{round(100 * quantile)}"] = {
            "trajectory_pass_rate_percent": score["trajectory_pass_rate_percent"],
            "cost_savings_score_percent": score["cost_savings_score_percent"],
            "combined_score_percent": score["combined_score_percent"],
            "under_route_percent": float(100 * np.mean(predictions < labels)),
            "selection_counts": dict(Counter(TIERS[value] for value in predictions)),
        }
    return result


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
    train_indices = np.array([index for index, row in enumerate(selected) if row["instance_id"] in train_ids])
    holdout_indices = np.array([index for index, row in enumerate(selected) if row["instance_id"] not in train_ids])
    train = [selected[index] for index in train_indices]
    holdout = [selected[index] for index in holdout_indices]
    labels = np.array([row["target_tier_id"] for row in selected], dtype=np.int64)
    train_y = labels[train_indices]
    holdout_y = labels[holdout_indices]
    groups = np.array([row["instance_id"] for row in train])

    embeddings, embedding_metadata = load_or_create_embeddings(
        selected,
        Path(args.embedding_cache),
        model_name=args.embedding_model,
        revision=args.embedding_revision,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    train_embeddings = embeddings[train_indices]
    holdout_embeddings = embeddings[holdout_indices]
    metadata = np.stack([metadata_features(row) for row in selected])
    jev_responses = _latest_responses(JournalClient(args.decomposed_journal, cap=0), "twinrouterbench-decomposed-v1")
    missing = [row["id"] for row in selected if row["id"] not in jev_responses]
    if missing:
        raise RuntimeError(f"missing decomposed Jev responses for {len(missing)} rows")
    compact_jev = np.stack([compact_jev_features(jev_responses[row["id"]]) for row in selected])
    full_aux = np.concatenate([metadata, compact_jev], axis=1)

    variants: dict[str, Any] = {}
    metadata_oof, metadata_folds = nested_oof_probabilities(metadata[train_indices], train_y, groups)
    variants["metadata_logistic_control"] = {
        "primary_nested_oof": probability_summary(train, train_y, metadata_oof, section11),
        "primary_posthoc_risk_sweep": risk_sweep(
            train, train_y, metadata_oof, section11
        ),
        "outer_folds": metadata_folds,
    }
    definitions: dict[str, np.ndarray | None] = {
        "embedding_knn": None,
        "embedding_metadata_knn": metadata,
        "embedding_metadata_compact_jev_knn": full_aux,
    }
    for name, aux in definitions.items():
        train_aux = None if aux is None else aux[train_indices]
        holdout_aux = None if aux is None else aux[holdout_indices]
        oof, folds = nested_knn_probabilities(train_embeddings, train_y, groups, train_aux)
        holdout_probabilities, search, fitted_train, fitted_holdout = fit_predict_knn(
            train_embeddings,
            train_y,
            groups,
            holdout_embeddings,
            train_aux,
            holdout_aux,
        )
        variants[name] = {
            "primary_nested_oof": probability_summary(train, train_y, oof, section11),
            "primary_posthoc_risk_sweep": risk_sweep(train, train_y, oof, section11),
            "outer_folds": folds,
            "final_training_search": search,
            "secondary_reused_holdout": probability_summary(
                holdout, holdout_y, holdout_probabilities, section11
            ),
            "secondary_posthoc_risk_sweep": risk_sweep(
                holdout, holdout_y, holdout_probabilities, section11
            ),
            "secondary_holdout_neighbors": neighbor_evidence(
                train, train_y, holdout, fitted_train, fitted_holdout
            ),
        }

    result = {
        "experiment": "twinrouterbench-semantic-knn-v1",
        "benchmark_commit": PINNED_TWINROUTERBENCH_COMMIT,
        "api_cost_usd": 0.0,
        "embedding": embedding_metadata,
        "split": split,
        "train_rows": len(train),
        "train_trajectories": len(set(groups)),
        "holdout_rows": len(holdout),
        "holdout_trajectories": len(set(row["instance_id"] for row in holdout)),
        "train_tier_counts": dict(Counter(TIERS[value] for value in train_y)),
        "evaluation_design": {
            "primary": "Five-fold trajectory-grouped nested evaluation; KNN hyperparameters are selected only inside each outer training fold.",
            "secondary": "Previously inspected holdout; diagnostic only.",
            "policy": "q90 selects the smallest tier covering 90% predicted required-tier probability.",
            "candidate_grid": {
                "k": list(K_VALUES),
                "weighting": list(WEIGHTINGS),
                "aux_weight": list(AUX_WEIGHTS),
                "dirichlet_smoothing_per_class": SMOOTHING,
            },
            "label_scope": "Required-tier labels for the fixed TwinRouterBench four-model pool; not transferable capability labels for arbitrary future models.",
        },
        "variants": variants,
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
    parser.add_argument("--decomposed-journal", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDER)
    parser.add_argument("--embedding-revision", default="main")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
