"""Quality evidence is independent of the recorded benchmark pass bit."""
import ast
import re

REQUIRED = ("required_behavior", "instruction_compliance", "regressions", "verification_integrity")


def acceptance(evidence):
    if any(e.get("status") == "not_met" and e.get("severity") in ("major", "blocker") for e in evidence.values()):
        return "unacceptable"
    if any(evidence.get(name, {}).get("status") != "met" for name in REQUIRED):
        return "indeterminate"
    return "acceptable"


def inspect_artifact(row):
    text = row["prediction"]
    if not isinstance(text, str):
        return {"parse": "indeterminate", "quality": "indeterminate", "reason": "Nontext prediction"}
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    code = blocks[-1] if blocks else text
    try:
        ast.parse(code)
        parses = True
    except SyntaxError:
        parses = False
    return {"parse": parses, "quality": "indeterminate", "recorded_test_pass": bool(row["score"]),
            "code_characters": len(code), "reason": "Parsing and archived tests do not establish full quality"}
