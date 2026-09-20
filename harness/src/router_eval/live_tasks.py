from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


@dataclass(frozen=True)
class LiveTask:
    task_id: str
    difficulty: str
    prompt: str
    editable: tuple[str, ...]
    root: Path

    @property
    def template(self):
        return self.root / "workspace"

    @property
    def hidden_tests(self):
        return self.root / "hidden_tests"

    def context(self):
        sections = [self.prompt, "\nRepository files:"]
        for path in sorted(self.template.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.template).as_posix()
                sections.append(f"\n--- {relative} ---\n{path.read_text(encoding='utf-8')}")
        return "\n".join(sections)

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update((self.root / "task.json").read_bytes())
        for path in sorted(self.root.rglob("*.py")):
            digest.update(path.relative_to(self.root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()


def load_tasks(root):
    tasks = []
    for manifest in sorted(Path(root).glob("*/task.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError("Unexpected task difficulty")
        tasks.append(LiveTask(data["id"], data["difficulty"], data["prompt"],
                              tuple(data["editable"]), manifest.parent))
    if len(tasks) != 6 or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("The frozen pilot requires six unique tasks")
    return tasks


class Workspace:
    def __init__(self, task, path):
        self.task = task
        self.path = Path(path)
        if not self.path.exists():
            shutil.copytree(task.template, self.path)

    def _resolve(self, name, editable=False):
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError("Unsafe workspace path")
        normalized = pure.as_posix()
        if editable and normalized not in self.task.editable:
            raise ValueError("File is not editable")
        target = (self.path / Path(*pure.parts)).resolve()
        if self.path.resolve() not in target.parents:
            raise ValueError("Path escapes workspace")
        return target

    def list_files(self):
        return [path.relative_to(self.path).as_posix() for path in sorted(self.path.rglob("*")) if path.is_file()]

    def read(self, name):
        target = self._resolve(name)
        if not target.is_file():
            raise ValueError("File does not exist")
        return target.read_text(encoding="utf-8")

    def write(self, name, content):
        if not isinstance(content, str) or len(content.encode()) > 40000:
            raise ValueError("File content is invalid or too large")
        target = self._resolve(name, editable=True)
        target.write_text(content, encoding="utf-8")
        return {"written": name, "bytes": len(content.encode())}
