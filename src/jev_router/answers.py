from .contracts import probabilities


def normalize(answers):
    # Choice supplies distributions; Noul supplies P(yes), not a distribution.
    value = answers["missing"]["noul"]
    probabilities({"yes": value, "no": 1 - value}, ("yes", "no"))
    return {"family": answers["family"]["probabilities"], "demand": answers["demand"]["probabilities"],
            "missing": {"yes": value, "no": 1 - value}}
