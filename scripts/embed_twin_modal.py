"""Generate the TwinRouterBench embedding cache on a Modal GPU."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import modal
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "harness" / "src")]

EMBEDDING_INSTRUCTION = (
    "Represent this coding-agent state for retrieval of prior states requiring similar "
    "reasoning capability and model strength."
)


app = modal.App("jev-router-twin-embeddings")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy>=1.26,<3",
    "sentence-transformers==5.7.0",
)
model_cache = modal.Volume.from_name("jev-router-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=1_800,
    volumes={"/root/.cache/huggingface": model_cache},
)
def embed(
    texts: list[str], model_name: str, revision: str, max_seq_length: int, batch_size: int
) -> tuple[bytes, str | None]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    model.max_seq_length = max_seq_length
    values = model.encode(
        texts,
        prompt=EMBEDDING_INSTRUCTION + "\nQuery: ",
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    config = getattr(getattr(model._first_module(), "auto_model", None), "config", None)
    resolved_revision = getattr(config, "_commit_hash", None)
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue(), resolved_revision


@app.local_entrypoint()
def main(
    benchmark_root: str,
    split_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    revision: str = "main",
    max_seq_length: int = 256,
    batch_size: int = 32,
) -> None:
    from router_eval.twin_calibration import _load_rows, _validate_split
    from router_eval.twin_knn import SERIALIZER_VERSION, _hash_text, serialize_state
    from router_eval.twinrouterbench import PINNED_TWINROUTERBENCH_COMMIT, _checkout_commit

    benchmark = Path(benchmark_root).resolve()
    if _checkout_commit(benchmark) != PINNED_TWINROUTERBENCH_COMMIT:
        raise RuntimeError("TwinRouterBench checkout differs from frozen commit")
    rows = _load_rows(benchmark / "data" / "static" / "question_bank.jsonl")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    _validate_split(split, rows)
    selected_ids = set(split["train"]) | set(split["holdout"])
    selected = [row for row in rows if row["instance_id"] in selected_ids]
    texts = [serialize_state(row) for row in selected]
    raw, resolved_revision = embed.remote(
        texts, model_name, revision, max_seq_length, batch_size
    )
    embeddings = np.load(io.BytesIO(raw), allow_pickle=False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, embeddings=embeddings)
    metadata = {
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "serializer_version": SERIALIZER_VERSION,
        "max_seq_length": max_seq_length,
        "ids": [str(row["id"]) for row in selected],
        "text_sha256": [_hash_text(text) for text in texts],
        "embedding_dimension": int(embeddings.shape[1]),
        "instruction": EMBEDDING_INSTRUCTION,
        "cache_format": "numpy-npz-v1",
        "compute": "Modal L4",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"rows": len(selected), "shape": list(embeddings.shape), **metadata}))
