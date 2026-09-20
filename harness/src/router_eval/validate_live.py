import json
from pathlib import Path
import shutil

from .live_tasks import Workspace, load_tasks
from .sandbox import DockerTests


def validate_live_tasks(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tests = DockerTests()
    results = {}
    for task in load_tasks(root / "harness/live_tasks"):
        bad_path = output / task.task_id / "seeded_bad"
        bad_workspace = Workspace(task, bad_path)
        bad_hidden = tests.run(bad_workspace.path, task.hidden_tests)
        good_path = output / task.task_id / "reference"
        good_workspace = Workspace(task, good_path)
        for editable in task.editable:
            shutil.copyfile(task.root / "reference" / editable, good_workspace.path / editable)
        good_visible = tests.run(good_workspace.path, "tests")
        good_hidden = tests.run(good_workspace.path, task.hidden_tests)
        results[task.task_id] = {
            "seeded_bad_rejected": not bad_hidden.passed,
            "reference_visible_pass": good_visible.passed,
            "reference_hidden_pass": good_hidden.passed,
            "bad_output": bad_hidden.output,
            "visible_output": good_visible.output,
            "hidden_output": good_hidden.output,
        }
        if bad_hidden.passed or not good_visible.passed or not good_hidden.passed:
            raise RuntimeError(f"Evaluator validation failed for {task.task_id}")
    (output / "validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
