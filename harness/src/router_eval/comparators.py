"""Published RouteLLM MF inference equations, implemented with NumPy.
Source attribution and pinned checkpoint live in routellm_manifest.json.
This is a port, not execution of the upstream PyTorch package.
"""
import json
import struct
from pathlib import Path
import numpy as np


def read_weights(path):
    raw = Path(path).read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    tensors = {}
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        if spec["dtype"] != "F32":
            raise ValueError("Expected float32 checkpoint")
        start, end = spec["data_offsets"]
        tensors[key] = np.frombuffer(raw[8 + size + start:8 + size + end], dtype="<f4").reshape(spec["shape"]).copy()
    return tensors


def mf_score(weights, embedding):
    embed = np.asarray(embedding, dtype=np.float32)
    p = weights["P.weight"][[24, 36]]
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    projected = weights["text_proj.0.weight"] @ embed
    logits = (p * projected) @ weights["classifier.0.weight"].T
    delta = float(logits[0, 0] - logits[1, 0])
    return float(1 / (1 + np.exp(-delta)))


def get_embeddings(tasks, client):
    result = client.post("/api/v1/embeddings",
        {"model": "openai/text-embedding-3-small", "input": [t.prompt for t in tasks], "encoding_format": "float"},
        "routellm-embeddings", reserve=0.02)
    rows = sorted(result["data"], key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(len(tasks))):
        raise ValueError("Embedding response indices do not match requests")
    array = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
    if array.shape != (len(tasks), 1536) or not np.isfinite(array).all():
        raise ValueError("Wrong or invalid embedding dimensions")
    return array
