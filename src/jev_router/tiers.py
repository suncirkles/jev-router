"""Three-tier B1 policy; no solver outcomes are visible at decision time."""
from dataclasses import dataclass

from .contracts import Decision, Task, digest, probabilities
from .policy import QUESTIONS


@dataclass(frozen=True)
class TierCandidates:
    economy: str
    standard: str
    frontier: str

    def __post_init__(self):
        if len({self.economy, self.standard, self.frontier}) != 3 or not all(
            (self.economy, self.standard, self.frontier)
        ):
            raise ValueError("Three distinct model tiers are required")


def decide_tier(task: Task, candidates: TierCandidates, assessments) -> Decision:
    probabilities(assessments["family"], QUESTIONS["family"]["criteria"])
    demand = probabilities(assessments["demand"], ("0", "1", "2"))
    missing = probabilities(assessments["missing"], ("yes", "no"))
    level = max((0, 1, 2), key=lambda value: (demand[str(value)], value))
    if missing["yes"] >= 0.5:
        level = 2
    model = (candidates.economy, candidates.standard, candidates.frontier)[level]
    return Decision(task.task_id, model, f"demand={level}; missing={missing['yes']:.3f}", digest(task.visible()))
