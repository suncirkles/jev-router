from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_FLOOR


def allocate(total_cents, weights):
    if isinstance(total_cents, bool) or not isinstance(total_cents, int):
        raise TypeError("total_cents must be an integer")
    if total_cents < 0:
        raise ValueError("total_cents must be non-negative")
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")
    converted = {}
    for key, raw in weights.items():
        if not isinstance(key, str) or not key:
            raise TypeError("weight keys must be non-empty strings")
        try:
            value = Decimal(raw)
        except (InvalidOperation, TypeError, ValueError):
            raise TypeError("weight values must be Decimal-compatible") from None
        if not value.is_finite() or value < 0:
            raise ValueError("weights must be finite and non-negative")
        converted[key] = value
    total_weight = sum(converted.values(), Decimal(0))
    if total_weight <= 0:
        raise ValueError("at least one weight must be positive")
    exact = {key: Decimal(total_cents) * converted[key] / total_weight for key in converted}
    result = {key: int(exact[key].to_integral_value(rounding=ROUND_FLOOR)) for key in sorted(converted)}
    remaining = total_cents - sum(result.values())
    ranked = sorted(converted, key=lambda key: (-(exact[key] - result[key]), key))
    for key in ranked[:remaining]:
        result[key] += 1
    return result
