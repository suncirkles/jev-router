"""A deliberately small file-editing loop shared by every model arm."""
import json
from pathlib import Path

from .live_tasks import Workspace


TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List repository files.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 repository file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace an editable UTF-8 file with complete content.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the visible unit tests in the isolated sandbox.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]

SYSTEM = """You are working in a small Python repository. Complete the requested change using the tools.
You may only edit the explicitly editable source files. Inspect files when needed, implement the complete behavior, and run visible tests.
Hidden tests will check edge cases from the written requirements. Do not merely describe code or return a patch in prose: use write_file.
Finish with a concise summary after testing. You have a bounded number of turns."""


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _assistant_message(message):
    return {key: message[key] for key in ("role", "content", "tool_calls", "reasoning_details") if key in message}


def _arguments(call):
    raw = call["function"].get("arguments", {})
    return json.loads(raw) if isinstance(raw, str) else raw


def run_agent(task, arm, model, run_root, client, tests, plugins=None, max_turns=7):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    workspace = Workspace(task, run_root / "workspace")
    state_path = run_root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "turn": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.context()}],
            "selected_models": [],
            "tool_events": [],
        }
        _save(state_path, state)
    while state["turn"] < max_turns:
        payload = {
            "model": model,
            "session_id": f"jev-route-live-002-{arm}-{task.task_id}",
            "messages": state["messages"],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 4096,
            "usage": {"include": True},
        }
        if plugins:
            payload["plugins"] = plugins
        reserve = 0.30 if model in {"openai/gpt-5.6-sol", "openrouter/auto"} else 0.12
        response = client.post("/api/v1/chat/completions", payload,
                               f"session:{arm}:{task.task_id}:turn:{state['turn']}", reserve)
        state["selected_models"].append(response.get("model"))
        message = response["choices"][0]["message"]
        state["messages"].append(_assistant_message(message))
        calls = message.get("tool_calls") or []
        state["turn"] += 1
        if not calls:
            _save(state_path, state)
            break
        for call in calls:
            name = call["function"]["name"]
            try:
                arguments = _arguments(call)
                if name == "list_files":
                    value = workspace.list_files()
                elif name == "read_file":
                    value = workspace.read(arguments["path"])
                elif name == "write_file":
                    value = workspace.write(arguments["path"], arguments["content"])
                elif name == "run_tests":
                    outcome = tests.run(workspace.path, "tests")
                    value = {"passed": outcome.passed, "returncode": outcome.returncode, "output": outcome.output[-6000:]}
                else:
                    raise ValueError("Unknown tool")
                event = {"tool": name, "ok": True, "result": value}
            except Exception as error:
                event = {"tool": name, "ok": False, "error": f"{type(error).__name__}: {error}"}
            state["tool_events"].append(event)
            state["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                      "content": json.dumps(event, ensure_ascii=False)})
        _save(state_path, state)
    visible = tests.run(workspace.path, "tests")
    hidden = tests.run(workspace.path, task.hidden_tests)
    completed_events = [event for event in client.events()
                        if event["status"] == "complete" and event["tag"].startswith(f"session:{arm}:{task.task_id}:")]
    result = {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "arm": arm,
        "requested_model": model,
        "selected_models": state["selected_models"],
        "turns": state["turn"],
        "tool_events": state["tool_events"],
        "visible_pass": visible.passed,
        "hidden_pass": hidden.passed,
        "visible_output": visible.output,
        "hidden_output": hidden.output,
        "cost": sum(event["cost"] for event in completed_events),
        "all_costs_measured": all(event["cost_measured"] for event in completed_events),
        "edited": {name: workspace.read(name) for name in task.editable},
    }
    _save(result_path, result)
    return result
