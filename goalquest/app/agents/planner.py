from __future__ import annotations

from ..llm import LLM
from ..ml.estimator import DurationCalibrator
from ..schemas import PlanSpec

PLANNER_SYSTEM = """You are the Planner for a gamified goal-achievement app.
Decompose the user's goal into Chapters -> Quests -> Tasks.
Rules:
- Work backwards from the deadline; total task minutes must fit weekly_hours * weeks * 0.8 (20% buffer).
- Every task is 10-90 minutes and has a VERIFIABLE done_when (a quiz, a commit, a solved set, a written summary).
- Each quest mixes kinds: learn, do, review.
- The LAST quest of every chapter is a boss quest (is_boss=true): an integrative challenge, e.g. a mock test or mini-project.
- Use depends_on to express prerequisites. Task keys must be unique. No cycles.
- Prefer small, concrete tasks a tired person could still start."""


def total_minutes(plan: PlanSpec) -> int:
    return sum(t.minutes for c in plan.chapters for q in c.quests for t in q.tasks)


def budget_minutes(weekly_hours: float, deadline_days: int, buffer: float = 0.2) -> float:
    """Minutes available for tasks, computed from what the USER asked for
    (never from values echoed back by the LLM)."""
    return weekly_hours * 60 * (deadline_days / 7) * (1 - buffer)


def generate_plan(llm: LLM, goal: str, why: str, deadline_days: int, weekly_hours: float,
                  context: list[str] | None = None,
                  calibrator: DurationCalibrator | None = None, attempts: int = 3) -> PlanSpec:
    ctx = "\n".join(f"- {c}" for c in (context or [])) or "(none)"
    user = (f"Goal: {goal}\nWhy it matters: {why}\nDeadline: {deadline_days} days\n"
            f"Weekly hours: {weekly_hours}\nWhat we know about this user:\n{ctx}")
    feedback = ""
    plan: PlanSpec | None = None
    for _ in range(attempts):
        plan = llm.structured(PLANNER_SYSTEM, user + feedback, PlanSpec)
        if calibrator:
            for c in plan.chapters:
                for q in c.quests:
                    for t in q.tasks:
                        t.minutes = min(90, calibrator.adjust(t.kind.value, t.minutes))
        plan.deadline_days, plan.weekly_hours = deadline_days, weekly_hours  # user input is authoritative
        used, avail = total_minutes(plan), budget_minutes(weekly_hours, deadline_days)
        if used <= avail:
            return plan
        feedback = (f"\n\nYour previous plan used {used} min but only {avail:.0f} min are available. "
                    f"Cut scope or merge tasks and re-emit.")
    assert plan is not None
    return plan  # best effort; caller can surface the over-budget warning
