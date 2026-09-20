def collapse(values):
    items = list(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in items):
        raise TypeError("all values must be integers")
    ordered = sorted(set(items))
    if not ordered:
        return ""
    result = []
    start = previous = ordered[0]
    for value in ordered[1:] + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        result.append(str(start) if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ",".join(result)
