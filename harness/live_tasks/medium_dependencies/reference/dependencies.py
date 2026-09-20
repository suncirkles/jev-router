from collections.abc import Iterable, Mapping
import heapq


class DependencyError(ValueError):
    pass


def build_order(dependencies):
    if not isinstance(dependencies, Mapping):
        raise DependencyError("dependencies must be a mapping")
    names = set(dependencies)
    if any(not isinstance(name, str) or not name for name in names):
        raise DependencyError("task names must be non-empty strings")
    normalized = {}
    for name, raw in dependencies.items():
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
            raise DependencyError("dependencies must be non-string iterables")
        deps = set(raw)
        if any(not isinstance(dep, str) or not dep for dep in deps):
            raise DependencyError("dependency names must be non-empty strings")
        if name in deps or not deps <= names:
            raise DependencyError("self or missing dependency")
        normalized[name] = deps
    dependents = {name: set() for name in names}
    for name, deps in normalized.items():
        for dep in deps:
            dependents[dep].add(name)
    ready = [name for name in names if not normalized[name]]
    heapq.heapify(ready)
    order = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for dependent in sorted(dependents[name]):
            normalized[dependent].remove(name)
            if not normalized[dependent]:
                heapq.heappush(ready, dependent)
    if len(order) != len(names):
        raise DependencyError("cycle detected")
    return order
