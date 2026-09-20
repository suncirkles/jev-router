import unicodedata


def normalize_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    parts = []
    pending_separator = False
    for character in normalized:
        if character.isalnum():
            if pending_separator and parts:
                parts.append("-")
            parts.append(character)
            pending_separator = False
        else:
            pending_separator = True
    result = "".join(parts)
    if not result:
        raise ValueError("value has no alphanumeric characters")
    return result
