import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, User
from app.schemas import PlanSpec


def make_plan(deadline_days=28, weekly_hours=6.0) -> PlanSpec:
    def t(k, title, kind, mins, diff=2, deps=()):
        return {"key": k, "title": title, "kind": kind, "minutes": mins, "difficulty": diff,
                "done_when": f"explain {title}", "depends_on": list(deps)}
    return PlanSpec.model_validate({
        "goal": "Learn linear algebra", "why": "to crack GATE DA", "deadline_days": deadline_days,
        "weekly_hours": weekly_hours,
        "chapters": [{"key": "c1", "title": "Vectors", "quests": [
            {"key": "q1", "title": "Basics", "tasks": [
                t("t1", "Vector spaces", "learn", 45, 1),
                t("t2", "Practice set", "do", 60, 2, ["t1"]),
                t("t3", "Review notes", "review", 20, 1, ["t2"])]},
            {"key": "q2", "title": "Boss: mock", "is_boss": True, "tasks": [
                t("t4", "Mock test", "do", 90, 3, ["t3"])]}]}]})


class FakeLLM:
    def __init__(self):
        self.calls = []

    def structured(self, system, user, schema, retries=2):
        self.calls.append(("structured", schema.__name__))
        if schema is PlanSpec:
            return make_plan()
        return schema.model_validate(
            {"passed": True, "score": 0.9, "feedback": "good"} if schema.__name__ == "VerificationResult"
            else {"questions": ["Explain it"], "rubric": "must mention basis"})

    def text(self, system, user, max_tokens=300):
        self.calls.append(("text", system[:30]))
        return "Keep going, you've got this!"

    def run_tool_loop(self, system, user, tools, handlers, max_turns=8):
        self.calls.append(("run_tool_loop", user))
        if "get_dashboard" in handlers:
            handlers["get_dashboard"]()
        return "Here is your progress update!"


@pytest.fixture
def db_factory():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    return sessionmaker(eng, expire_on_commit=False)


@pytest.fixture
def user_id(db_factory):
    with db_factory() as db:
        u = User(name="Harish", coach_style="mentor")
        db.add(u)
        db.commit()
        return u.id


@pytest.fixture
def fake_llm():
    return FakeLLM()
