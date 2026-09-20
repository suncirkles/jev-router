"""Single-writer, durable request journal with conservative spend reservation."""
import json
import math
import os
from pathlib import Path
import requests
from .contracts import digest


class JournalClient:
    def __init__(self, path, cap=0.50, timeout=180):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = self._amount(cap, "cap")
        self.timeout = timeout

    @staticmethod
    def _amount(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return float(value)

    def events(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def append(self, event):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def accounted(self):
        latest = {}
        for event in self.events():
            latest[event["key"]] = event
        return sum(e.get("cost", e["reserve"]) for e in latest.values())

    def post(self, endpoint, payload, tag, reserve):
        reserve = self._amount(reserve, "reserve")
        key = digest({"endpoint": endpoint, "payload": payload, "tag": tag})
        latest = {}
        for event in self.events():
            latest[event["key"]] = event
        if any(e["status"] in {"pending", "needs_review"} for e in latest.values()):
            raise RuntimeError("Unresolved request: reconcile the journal before sending more paid calls")
        if key in latest and latest[key]["status"] == "complete":
            return latest[key]["response"]
        if key in latest:
            raise RuntimeError("Previously failed request; inspect error before explicit retry")
        if self.accounted() + reserve > self.cap:
            raise RuntimeError("Pilot spend cap reached before dispatch")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        event = {"key": key, "tag": tag, "reserve": reserve, "status": "pending"}
        self.append(event)
        # Timeouts intentionally leave a pending entry: do not blindly rebill.
        response = requests.post("https://openrouter.ai" + endpoint, json=payload,
                                 headers={"Authorization": "Bearer " + api_key,
                                          "X-OpenRouter-Metadata": "enabled"},
                                 timeout=self.timeout)
        if response.status_code != 200:
            self.append({**event, "status": "failed", "http_status": response.status_code,
                         "cost": reserve})
            raise RuntimeError(f"OpenRouter HTTP {response.status_code}; reserved charge retained")
        body = response.json()
        if "error" in body:
            self.append({**event, "status": "failed", "cost": reserve})
            raise RuntimeError("Provider returned an error; reserved charge retained")
        safe = {k: body[k] for k in ("id", "model", "answers", "choices", "data", "usage", "provider", "openrouter_metadata") if k in body}
        usage = body.get("usage", {})
        measured = "cost" in usage
        cost = self._amount(usage["cost"], "provider cost") if measured else reserve
        if cost > reserve:
            self.append({**event, "status": "needs_review", "cost": cost,
                         "cost_measured": measured, "response": safe})
            raise RuntimeError("Provider cost exceeded reservation; stopped for review")
        self.append({**event, "status": "complete", "cost": cost,
                     "cost_measured": measured, "response": safe})
        return safe
