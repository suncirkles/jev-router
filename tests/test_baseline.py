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
from router_eval.live_runner import _cheapest_passing_tier, _frozen_cap
from router_eval.metrics import evaluate, paired_interval
from router_eval.quality import acceptance, inspect_artifact, REQUIRED
from router_eval.sandbox import DockerTests
from router_eval.twinrouterbench import (
    JevTwinPredictor,
    TIER_QUESTION,
    opaque_model_mapping,
    opaque_tier_question,
    tier_probabilities,
    visible_state,
)
from router_eval.twin_calibration import (
    features as calibration_features,
    metadata_features,
    quantile_tiers,
)
from router_eval.twin_decomposition import (
    DECOMPOSED_FEATURE_NAMES,
    DECOMPOSED_QUESTIONS,
    decomposed_features,
    nested_oof_probabilities,
)
from router_eval.twin_knn import (
    compact_jev_features,
    knn_probabilities,
    nested_knn_probabilities,
    serialize_state,
)

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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_money_values_are_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        JournalClient(tmp_path / "journal", cap=value)


def test_over_reservation_remains_blocked_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")

    class Response:
        status_code = 200

        def json(self):
            return {"model": "example", "usage": {"cost": .003}}

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Response())
    journal = tmp_path / "journal"
    with pytest.raises(RuntimeError, match="exceeded reservation"):
        JournalClient(journal).post("/api/test", {}, "test", .002)
    with pytest.raises(RuntimeError, match="Unresolved"):
        JournalClient(journal).post("/api/test", {}, "test", .002)


def test_live_metrics_use_observed_cost_and_frozen_cap():
    fixed = {
        "economy": {"hidden_pass": True, "cost": .1},
        "standard": {"hidden_pass": True, "cost": .02},
        "frontier": {"hidden_pass": False, "cost": .01},
    }
    assert _cheapest_passing_tier(fixed) == "standard"
    assert _frozen_cap({"spend_cap": 2.0}, 2.0) == 2.0
    with pytest.raises(RuntimeError, match="frozen"):
        _frozen_cap({"spend_cap": 2.0}, 3.0)


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


def test_docker_timeout_bytes_are_decoded(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise __import__("subprocess").TimeoutExpired(
            "docker", 20, output=b"partial \xce\xb1", stderr=b"error"
        )

    monkeypatch.setattr("subprocess.run", timeout)
    result = DockerTests().run(tmp_path)
    assert result.returncode == 124
    assert "partial αerror" in result.output
    assert result.output.endswith("TIMEOUT")


def test_twinrouter_visible_state_excludes_labels():
    row = {
        "id": "step-1",
        "benchmark": "swebench",
        "scenario": "code_swe",
        "instance_id": "repo-1",
        "step_index": 2,
        "total_steps": 4,
        "messages": [{"role": "user", "content": "fix it"}],
        "target_tier": "high",
        "target_tier_id": 3,
        "notes": "private evaluator metadata",
    }
    state = visible_state(row)
    assert state["messages"] == row["messages"]
    assert "target_tier" not in state
    assert "target_tier_id" not in state
    assert "notes" not in state


def test_knn_state_serialization_is_router_visible_and_keeps_recent_evidence():
    row = {
        "id": "secret-label-must-not-appear",
        "benchmark": "swebench",
        "scenario": "code_swe",
        "step_index": 3,
        "total_steps": 5,
        "target_tier": "high",
        "target_tier_id": 3,
        "messages": [
            {"role": "user", "content": "fix the cache"},
            {"role": "assistant", "content": "x" * 100},
            {"role": "tool", "content": "LATEST FAILURE"},
        ],
    }
    rendered = serialize_state(row, original_limit=20, recent_limit=80)
    assert "fix the cache" in rendered
    assert "LATEST FAILURE" in rendered
    assert "target_tier" not in rendered
    assert "secret-label-must-not-appear" not in rendered


def test_compact_jev_features_are_expected_normalized_severities():
    answers = {
        key: {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.5}}
        for key in DECOMPOSED_QUESTIONS
    }
    assert compact_jev_features({"answers": answers}).tolist() == pytest.approx([0.625] * 5)


def test_knn_probabilities_prefer_nearest_label_and_remain_calibratable():
    train = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = np.array([0, 2, 3])
    query = np.array([[0.99, 0.01]])
    probabilities = knn_probabilities(
        train, labels, query, k=2, weighting="distance", smoothing=0.25
    )
    assert probabilities.shape == (1, 4)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0])
    assert int(np.argmax(probabilities[0])) == 0
    assert np.all(probabilities > 0)


def test_nested_knn_returns_one_probability_row_per_input():
    embeddings = np.eye(20, dtype=np.float64)
    labels = np.array([index % 4 for index in range(20)])
    groups = np.array([f"trajectory-{index // 2}" for index in range(20)])
    probabilities, folds = nested_knn_probabilities(embeddings, labels, groups)
    assert probabilities.shape == (20, 4)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0] * 20)
    assert len(folds) == 5


def test_twinrouter_jev_selects_highest_probability_and_conservative_tie():
    row = {"id": "x", "benchmark": "swebench", "messages": []}

    class Client:
        def __init__(self, values):
            self.values = values
            self.payload = None

        def post(self, endpoint, payload, tag, reserve):
            self.payload = payload
            return {
                "model": "typesafe/jev-1.13-20260917",
                "answers": {"tier": {"probabilities": self.values}},
            }

    client = Client({"0": 0.1, "1": 0.4, "2": 0.4, "3": 0.1})
    assert JevTwinPredictor(client).predict(row).tier_id == 2
    assert client.payload["questions"] == TIER_QUESTION
    assert "target_tier_id" not in client.payload["state"]


def test_opaque_question_hides_model_names_and_remaps_probabilities():
    mapping = opaque_model_mapping(20260922)
    assert mapping == (2, 0, 3, 1)
    question = opaque_tier_question(mapping)
    serialized = json.dumps(question)
    assert all(name not in serialized for name in ("DeepSeek", "MiniMax", "Gemini", "Claude"))
    assert "Model A" in serialized and "Model D" in serialized
    assert "relative cost is third-lowest" in question["tier"]["criteria"]["0"]
    assert "relative cost is lowest" in question["tier"]["criteria"]["1"]

    response = {
        "answers": {"tier": {"probabilities": {"0": .1, "1": .2, "2": .6, "3": .1}}}
    }
    assert tier_probabilities(response, mapping) == [.2, .1, .1, .6]


def test_opaque_predictor_uses_harness_mapping_and_separate_journal_namespace():
    row = {"id": "x", "benchmark": "swebench", "messages": []}
    mapping = (2, 0, 3, 1)

    class Client:
        def __init__(self):
            self.tag = None

        def post(self, endpoint, payload, tag, reserve):
            self.tag = tag
            return {
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "tier": {"probabilities": {"0": .1, "1": .2, "2": .6, "3": .1}}
                },
            }

    client = Client()
    predictor = JevTwinPredictor(
        client,
        question=opaque_tier_question(mapping),
        option_to_tier=mapping,
        tag_namespace="opaque-test",
    )
    assert predictor.predict(row).tier_id == 3
    assert client.tag == "opaque-test:x"


def test_calibration_features_do_not_depend_on_target_label():
    row = {
        "id": "x",
        "benchmark": "swebench",
        "messages": [{"role": "user", "content": "fix it"}],
        "step_index": 1,
        "total_steps": 2,
        "target_tier_id": 0,
    }
    response = {"answers": {"tier": {"probabilities": {"0": .6, "1": .3, "2": .08, "3": .02}}}}
    first = calibration_features(row, response)
    row["target_tier_id"] = 3
    assert np.array_equal(first, calibration_features(row, response))


def test_calibrated_quantile_routes_up_with_uncertainty():
    matrix = np.array([[.6, .2, .1, .1], [.05, .05, .1, .8]])
    assert quantile_tiers(matrix, .5).tolist() == [0, 3]
    assert quantile_tiers(matrix, .9).tolist() == [2, 3]


def test_decomposed_questions_and_features_do_not_expose_targets():
    serialized = json.dumps(DECOMPOSED_QUESTIONS)
    assert all(
        name not in serialized
        for name in ("DeepSeek", "MiniMax", "Gemini", "Claude", "low tier", "high tier")
    )
    assert set(DECOMPOSED_QUESTIONS) == {
        "scope",
        "uncertainty",
        "recovery",
        "correctness_risk",
        "verification_burden",
    }
    row = {
        "id": "x",
        "benchmark": "swebench",
        "messages": [{"role": "user", "content": "fix it"}],
        "step_index": 1,
        "total_steps": 3,
        "target_tier_id": 0,
    }
    response = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            key: {"probabilities": {"0": .6, "1": .3, "2": .1}}
            for key in DECOMPOSED_QUESTIONS
        },
    }
    first = decomposed_features(row, response)
    assert len(first) == len(DECOMPOSED_FEATURE_NAMES) + len(metadata_features(row))
    row["target_tier_id"] = 3
    assert np.array_equal(first, decomposed_features(row, response))


def test_nested_oof_probabilities_cover_each_row_once():
    labels = np.tile(np.arange(4), 10)
    groups = np.repeat(np.arange(10), 4)
    values = np.eye(4)[labels]
    matrix, folds = nested_oof_probabilities(values, labels, groups)
    assert matrix.shape == (40, 4)
    assert np.allclose(matrix.sum(axis=1), 1)
    held_out = [group for fold in folds for group in fold["held_out_trajectories"]]
    assert sorted(held_out) == list(range(10))
