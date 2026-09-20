"""The router accepts only information available before execution."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    language: str = "python"

    def __post_init__(self):
        if not self.task_id or not self.prompt.strip():
            raise ValueError("Task id and prompt are required")
        if len(self.prompt.encode()) > 24000:
            raise ValueError("Prompt exceeds pilot limit; never silently truncate")

    def visible(self):
        return asdict(self)


@dataclass(frozen=True)
class Candidates:
    cheap: str
    strong: str

    def __post_init__(self):
        if not self.cheap or not self.strong or self.cheap == self.strong:
            raise ValueError("Two distinct candidates are required")


@dataclass(frozen=True)
class Decision:
    task_id: str
    model: str
    reason: str
    input_hash: str
    fallback: bool = False


def probabilities(values, keys):
    if not isinstance(values, dict) or set(values) != set(keys):
        raise ValueError("Unexpected probability labels")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Invalid probability")
    if abs(sum(values.values()) - 1) > 0.031:
        raise ValueError("Probabilities must sum to one within rounding tolerance")
    return values
