"""Jev adapter. Provider details are isolated from routing policy."""
from .policy import QUESTIONS

MODEL = "~typesafe/jev-latest"
EXPECTED_MODEL = "typesafe/jev-1.13-20260917"


def classify(task, client):
    result = client.post("/api/alpha/decisions",
        {"model": MODEL, "state": task.visible(), "questions": QUESTIONS},
        "jev:" + task.task_id, reserve=0.003)
    if result.get("model") != EXPECTED_MODEL:
        raise RuntimeError("Jev resolved version changed; start a separately versioned experiment")
    return result
