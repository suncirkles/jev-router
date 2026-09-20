import json
import math
from pathlib import Path
import numpy as np
import pytest
from jev_router.answers import normalize
from jev_router.contracts import Task, Candidates, probabilities
from jev_router.policy import decide
from jev_router.provider import JournalClient
from jev_router.tiers import TierCandidates, decide_tier
from router_eval.comparators import mf_score, read_weights
from router_eval.data import MODELS, load, split, task_for, freeze
from router_eval.live_tasks import Workspace, load_tasks
from router_eval.metrics import evaluate, paired_interval
from router_eval.quality import acceptance, inspect_artifact, REQUIRED
from router_eval.sandbox import DockerTests

ROOT = Path(__file__).resolve().parents[1]


def assessment(level=0, missing=0):
    return {"family": {"implementation": 1, "debugging": 0, "refactor": 0, "explain": 0, "other": 0},
            "demand": {str(i): int(i == level) for i in range(3)}, "missing": {"yes": missing, "no": 1 - missing}}


@pytest.mark.parametrize("level,missing,expected", [(0, 0, "cheap"), (1, 0, "strong"), (2, 0, "strong"), (0, .5, "strong"), (0, .49, "cheap")])
def test_policy_boundary(level, missing, expected):
    assert decide(Task("x", "Task"), Candidates("cheap", "strong"), assessment(level, missing)).model == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 2, True])
def test_bad_probabilities_fail(bad):
    with pytest.raises(ValueError):
        probabilities({"yes": bad, "no": 0}, ("yes", "no"))


def test_missing_labels_and_wrong_sum_fail():
    with pytest.raises(ValueError):
        probabilities({"yes": .4}, ("yes", "no"))
    with pytest.raises(ValueError):
        probabilities({"yes": .4, "no": .4}, ("yes", "no"))


def test_no_truncation_and_distinct_candidates():
    with pytest.raises(ValueError):
        Task("id", "x" * 24001)
    with pytest.raises(ValueError):
        Candidates("x", "x")


def test_noul_normalization():
    a = assessment()
    response = {"family": {"probabilities": a["family"]}, "demand": {"probabilities": a["demand"]}, "missing": {"noul": .25}}
    assert normalize(response)["missing"] == {"yes": .25, "no": .75}


@pytest.mark.parametrize("level,missing,expected", [
    (0, 0, "economy"), (1, 0, "standard"), (2, 0, "frontier"), (0, .5, "frontier")
])
def test_three_tier_policy(level, missing, expected):
    candidates = TierCandidates("economy", "standard", "frontier")
    assert decide_tier(Task("x", "Task"), candidates, assessment(level, missing)).model == expected


def test_three_tier_candidates_are_distinct():
    with pytest.raises(ValueError):
        TierCandidates("same", "same", "frontier")


def test_live_tasks_frozen_and_workspace_confined(tmp_path):
    tasks = load_tasks(ROOT / "harness/live_tasks")
    assert [task.difficulty for task in tasks].count("easy") == 2
    assert [task.difficulty for task in tasks].count("medium") == 2
    assert [task.difficulty for task in tasks].count("hard") == 2
    task = tasks[0]
    workspace = Workspace(task, tmp_path / "workspace")
    with pytest.raises(ValueError):
        workspace.read("../task.json")
    with pytest.raises(ValueError):
        workspace.write("tests/test_visible.py", "pass")
    editable = task.editable[0]
    workspace.write(editable, "# changed\n")
    assert workspace.read(editable) == "# changed\n"


def test_quality_major_cannot_be_compensated():
    evidence = {k: {"status": "met"} for k in REQUIRED}
    assert acceptance(evidence) == "acceptable"
    evidence["security"] = {"status": "not_met", "severity": "blocker"}
    assert acceptance(evidence) == "unacceptable"


def test_missing_quality_and_parse_are_not_acceptance():
    assert acceptance({"required_behavior": {"status": "met"}}) == "indeterminate"
    assert inspect_artifact({"prediction": "print(42)", "score": 1})["quality"] == "indeterminate"
    assert inspect_artifact({"prediction": "def broken(", "score": 0})["parse"] is False


def test_mf_equations_against_independent_scalar_formula():
    rng = np.random.default_rng(42)
    p, projection, classifier, embed = [rng.normal(size=s).astype(np.float32) for s in [(64, 4), (4, 6), (1, 4), (6,)]]
    weights = {"P.weight": p, "text_proj.0.weight": projection, "classifier.0.weight": classifier}
    def logit(row):
        norm = math.sqrt(sum(float(x) ** 2 for x in p[row]))
        return sum(float(classifier[0, j]) * float(p[row, j]) / norm * sum(float(projection[j, i]) * float(embed[i]) for i in range(6)) for j in range(4))
    expected = 1 / (1 + math.exp(-(logit(24) - logit(36))))
    assert mf_score(weights, embed) == pytest.approx(expected, abs=1e-6)


def test_published_checkpoint_shape():
    weights = read_weights(ROOT / ".cache/model.safetensors")
    assert weights["P.weight"].shape == (64, 128)
    assert weights["text_proj.0.weight"].shape == (128, 1536)
    assert mf_score(weights, np.zeros(1536)) == .5


def test_frozen_split_and_runtime_have_no_outcomes():
    records, _ = load(ROOT / ".cache")
    selected = split(records)
    assert len(selected["calibration"]) == 20
    assert len(selected["evaluation"]) == 40
    assert not set(selected["calibration"]) & set(selected["evaluation"])
    assert selected == split(records)
    task = task_for(records, selected["evaluation"][0])
    assert set(task.visible()) == {"task_id", "prompt", "language"}
    # Flipping every outcome must not change selection.
    for rows in records.values():
        for row in rows.values():
            row["score"] = 1 - row["score"]
    assert selected == split(records)


def test_freeze_rejects_manifest_change(tmp_path):
    p = tmp_path / "f.json"
    freeze(p, {"seed": 1})
    freeze(p, {"seed": 1})
    with pytest.raises(RuntimeError):
        freeze(p, {"seed": 2})


def test_counterfactual_failure_and_zero_pass_cost():
    records = {"gemini-2.5-flash": {"x": {"score": 0, "cost": .1}}, "gpt-5": {"x": {"score": 1, "cost": 1}}}
    result = evaluate(["x"], {"x": "gemini-2.5-flash"}, records)
    assert result["avoidable_test_failures"] == 1
    assert result["cost_per_recorded_pass"] is None
    assert result["quality_acceptance_rate"] is None
    assert paired_interval([0, 1], [0, 1])["ci95"] == [0, 0]


def test_cap_and_pending_prevent_dispatch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network should not be called")
    monkeypatch.setattr("requests.post", forbidden)
    client = JournalClient(tmp_path / "journal", cap=.01)
    with pytest.raises(RuntimeError, match="cap"):
        client.post("/api/test", {}, "test", .02)
    client.append({"key": "prior", "status": "pending", "reserve": .001})
    with pytest.raises(RuntimeError, match="Unresolved"):
        client.post("/api/test", {}, "test", .001)


def test_paid_request_cache_is_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    class Response:
        status_code = 200
        def json(self):
            return {"model": "example", "usage": {"cost": .001}, "user_id": "do-not-store"}
    calls = []
    monkeypatch.setattr("requests.post", lambda *a, **kw: calls.append(1) or Response())
    client = JournalClient(tmp_path / "journal")
    a = client.post("/api/test", {}, "test", .002)
    b = client.post("/api/test", {}, "test", .002)
    assert a == b and len(calls) == 1
    assert client.accounted() == .001
    assert "user_id" not in (tmp_path / "journal").read_text()


def test_docker_output_is_decoded_as_utf8(tmp_path, monkeypatch):
    observed = {}

    class Completed:
        returncode = 0
        stdout = "Καλημέρα"
        stderr = None

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)
    result = DockerTests().run(tmp_path)
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"
    assert result.output == "Καλημέρα"
