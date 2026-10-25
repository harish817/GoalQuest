"""SQLAlchemy 2.0 models. The XP ledger is append-only: balances are always
derived by summing it, which makes rewards auditable and tamper-evident."""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    coach_style: Mapped[str] = mapped_column(String(30), default="mentor")
    streak: Mapped[int] = mapped_column(Integer, default=0)
    freezes: Mapped[int] = mapped_column(Integer, default=1)
    last_active: Mapped[date | None] = mapped_column(Date, nullable=True)
    goals: Mapped[list["Goal"]] = relationship(back_populates="user")


class Goal(Base):
    __tablename__ = "goals"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(200))
    why: Mapped[str] = mapped_column(Text, default="")
    deadline: Mapped[date | None] = mapped_column(Date, nullable=True)
    plan_json: Mapped[dict] = mapped_column(JSON, default=dict)
    user: Mapped[User] = relationship(back_populates="goals")
    tasks: Mapped[list["Task"]] = relationship(back_populates="goal")


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    goal_id: Mapped[int] = mapped_column(ForeignKey("goals.id"))
    key: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(10))
    est_minutes: Mapped[int] = mapped_column(Integer)
    difficulty: Mapped[int] = mapped_column(Integer, default=2)
    done_when: Mapped[str] = mapped_column(Text, default="")
    depends_on: Mapped[list] = mapped_column(JSON, default=list)
    is_boss: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(12), default="todo")  # todo|active|done
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    goal: Mapped[Goal] = relationship(back_populates="tasks")


class XPLedger(Base):
    __tablename__ = "xp_ledger"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    amount: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(200))
    day: Mapped[date] = mapped_column(Date, default=lambda: _now().date())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class BadgeAward(Base):
    __tablename__ = "badges"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    code: Mapped[str] = mapped_column(String(50))
    awarded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class CheckInRow(Base):
    __tablename__ = "checkins"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    day: Mapped[date] = mapped_column(Date, default=lambda: _now().date())
    energy: Mapped[int] = mapped_column(Integer)
    mood: Mapped[int] = mapped_column(Integer)
    minutes_focused: Mapped[int] = mapped_column(Integer, default=0)
    distractions: Mapped[int] = mapped_column(Integer, default=0)
    tasks_done: Mapped[int] = mapped_column(Integer, default=0)
    note: Mapped[str] = mapped_column(Text, default="")


class BanditArm(Base):
    """Per-user Beta posterior for each coaching-message style."""
    __tablename__ = "bandit_arms"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    arm: Mapped[str] = mapped_column(String(40))
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db(eng=engine) -> None:
    Base.metadata.create_all(eng)
    with sessionmaker(eng, expire_on_commit=False)() as db:
        seed_default_data(db)


def seed_default_data(db) -> None:
    user = db.get(User, 1)
    if not user:
        user = User(id=1, name="Harish", coach_style="mentor", streak=1, freezes=1)
        db.add(user)
        db.commit()

    goal = db.get(Goal, 1)
    if not goal:
        from . import service
        from .schemas import PlanSpec
        plan = PlanSpec.model_validate({
            "goal": "Master Machine Learning Foundations",
            "why": "Build intelligent systems and crack GATE DA",
            "deadline_days": 28,
            "weekly_hours": 6.0,
            "chapters": [{
                "key": "c1",
                "title": "Linear Algebra & Vectors",
                "quests": [
                    {
                        "key": "q1",
                        "title": "Vector Spaces & Foundations",
                        "tasks": [
                            {"key": "t1", "title": "Vector Spaces & Subspaces", "kind": "learn", "minutes": 45, "difficulty": 1, "done_when": "Explain vector space axioms and spanning sets", "depends_on": []},
                            {"key": "t2", "title": "Basis & Dimension Practice", "kind": "do", "minutes": 60, "difficulty": 2, "done_when": "Solve 5 linear independence and basis questions", "depends_on": ["t1"]},
                            {"key": "t3", "title": "Review Rank-Nullity Theorem", "kind": "review", "minutes": 20, "difficulty": 1, "done_when": "Summarize rank-nullity with 2 worked examples", "depends_on": ["t2"]}
                        ]
                    },
                    {
                        "key": "q2",
                        "title": "Boss Challenge: Matrix Transformations",
                        "is_boss": True,
                        "tasks": [
                            {"key": "t4", "title": "Boss: Matrix Transformation Mastery", "kind": "do", "minutes": 90, "difficulty": 3, "done_when": "Achieve >= 80% on 10 comprehensive matrix problems", "depends_on": ["t3"]}
                        ]
                    }
                ]
            }]
        })
        service.save_plan(db, user.id, plan)

