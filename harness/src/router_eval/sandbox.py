"""Run generated Python only inside a locked-down Docker container."""
from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class TestResult:
    passed: bool
    returncode: int
    output: str


IMAGE = "python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9"


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


class DockerTests:
    def __init__(self, image=IMAGE, timeout=20):
        self.image = image
        self.timeout = timeout

    def run(self, workspace, tests="tests"):
        workspace = Path(workspace).resolve()
        command = [
            "docker", "run", "--rm", "--network", "none", "--memory", "256m",
            "--cpus", "1", "--pids-limit", "64", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--volume", f"{workspace}:/workspace:ro", "--workdir", "/workspace",
        ]
        if Path(tests).is_absolute():
            tests = Path(tests).resolve()
            command += ["--volume", f"{tests}:/hidden:ro"]
            suite = "/hidden"
        else:
            suite = str(tests).replace("\\", "/")
        command += [self.image, "python", "-m", "unittest", "discover", "-s", suite, "-v"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=self.timeout)
            output = ((completed.stdout or "") + (completed.stderr or ""))[-12000:]
            return TestResult(completed.returncode == 0, completed.returncode, output)
        except subprocess.TimeoutExpired as error:
            output = (_text(error.stdout) + _text(error.stderr))[-12000:]
            return TestResult(False, 124, output + "\nTIMEOUT")
