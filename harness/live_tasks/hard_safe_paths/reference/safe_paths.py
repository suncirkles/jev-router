from pathlib import Path
import re


class UnsafePathError(ValueError):
    pass


def build_extraction_plan(names, destination: Path):
    destination = Path(destination)
    result = []
    seen = set()
    for original in names:
        if not isinstance(original, str) or not original or "\x00" in original:
            raise UnsafePathError("invalid member name")
        if original[0] in "/\\" or original[-1] in "/\\" or re.match(r"^[A-Za-z]:", original):
            raise UnsafePathError("absolute, drive, or directory path")
        components = []
        for component in original.replace("\\", "/").split("/"):
            if component in ("", "."):
                continue
            if component == "..":
                raise UnsafePathError("parent traversal")
            components.append(component)
        if not components:
            raise UnsafePathError("empty target")
        target = destination.joinpath(*components)
        collision_key = tuple(part.casefold() for part in target.parts)
        if collision_key in seen:
            raise UnsafePathError("target collision")
        seen.add(collision_key)
        result.append((original, target))
    return result
