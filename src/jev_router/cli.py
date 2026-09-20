import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .contracts import Task, Candidates
from .provider import JournalClient
from .classifier import classify
from .policy import decide
from .answers import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--journal", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    task = Task(**data["task"])
    result = classify(task, JournalClient(args.journal))
    print(json.dumps(asdict(decide(task, Candidates(**data["candidates"]), normalize(result["answers"])))))


if __name__ == "__main__":
    main()
