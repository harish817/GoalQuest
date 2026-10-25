"""Pydantic models. These double as the JSON schemas the LLM must satisfy
(structured output via Anthropic tool use), so a malformed plan is rejected
at the boundary instead of leaking into the database."""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class TaskKind(str, Enum):
    learn = "learn"
    do = "do"
    review = "review"


class Difficulty(int, Enum):
    easy = 1
    medium = 2
    hard = 3
    boss = 5


class TaskSpec(BaseModel):
    key: str = Field(description="Short unique id within the plan, e.g. 'c1-q2-t3'")
    title: str
    kind: TaskKind
    minutes: int = Field(ge=10, le=90, description="Estimated minutes; must be <= 90")
    difficulty: Difficulty = Difficulty.medium
    done_when: str = Field(description="Verifiable definition of done")
    depends_on: list[str] = Field(default_factory=list)


class QuestSpec(BaseModel):
    key: str
    title: str
    is_boss: bool = False
    tasks: list[TaskSpec] = Field(min_length=1)


class ChapterSpec(BaseModel):
    key: str
    title: str
    quests: list[QuestSpec] = Field(min_length=1)


class PlanSpec(BaseModel):
    goal: str
    why: str
    deadline_days: int = Field(gt=0)
    weekly_hours: float = Field(gt=0)
    chapters: list[ChapterSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dag(self) -> "PlanSpec":
        """Dependencies must reference known tasks and contain no cycles."""
        keys = [t.key for c in self.chapters for q in c.quests for t in q.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate task keys in plan")
        known = set(keys)
        deps = {t.key: t.depends_on for c in self.chapters for q in c.quests for t in q.tasks}
        for k, ds in deps.items():
            for d in ds:
                if d not in known:
                    raise ValueError(f"task {k} depends on unknown task {d}")
        # Kahn's algorithm for cycle detection
        indeg = {k: len(ds) for k, ds in deps.items()}
        ready = [k for k, n in indeg.items() if n == 0]
        seen = 0
        while ready:
            n = ready.pop()
            seen += 1
            for k, ds in deps.items():
                if n in ds:
                    indeg[k] -= 1
                    if indeg[k] == 0:
                        ready.append(k)
        if seen != len(keys):
            raise ValueError("plan dependencies contain a cycle")
        return self


class CheckIn(BaseModel):
    energy: int = Field(ge=1, le=5)
    mood: int = Field(ge=1, le=5)
    minutes_focused: int = Field(ge=0, le=600, default=0)
    distractions: int = Field(ge=0, default=0)
    note: str = ""


class VerificationResult(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    feedback: str
