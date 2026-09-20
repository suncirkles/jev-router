from pathlib import Path


class UnsafePathError(ValueError):
    pass


def build_extraction_plan(names, destination: Path):
    raise NotImplementedError
