def collapse(values):
    values = list(values)
    if not values:
        return ""
    return ",".join(str(value) for value in values)
