"""LangGraph daily-loop workflow.

   assess_risk --(high)----> recovery ---\
        |--(behind)--------> replan   ----+--> coach --> END
        '--(on track)------> normal   ---/

The graph holds *control flow*; the heavy lifting lives in service.py (rules),
ml/ (models) and agents/ (LLM calls).
"""
from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import service
from .agents import coach
from .db import BanditArm, Goal, User
from .game import reward_engine as re_
from .llm import LLM
from .ml.bandit import ARMS, ThompsonBandit, allowed_arms_for_state
from .ml.dropout import DropoutModel, risk_band
from .schemas import CheckIn


class DayState(TypedDict, total=False):
    user_id: int
    goal_id: int
    checkin: dict
    risk: float
    band: str
    ahead: float
    quests: list[dict]
    arm: str
    message: str
    tips: list[str]
    replan_deadline: str | None


def load_bandit(db: Session, user_id: int, seed: int | None = None) -> ThompsonBandit:
    b = ThompsonBandit(seed=seed)
    for row in db.scalars(select(BanditArm).where(BanditArm.user_id == user_id)).all():
        if row.arm in b.alpha:
            b.alpha[row.arm], b.beta[row.arm] = row.alpha, row.beta
    return b


def save_bandit(db: Session, user_id: int, b: ThompsonBandit) -> None:
    existing = {r.arm: r for r in db.scalars(select(BanditArm).where(BanditArm.user_id == user_id)).all()}
    for arm in ARMS:
        row = existing.get(arm) or BanditArm(user_id=user_id, arm=arm)
        row.alpha, row.beta = b.alpha[arm], b.beta[arm]
        db.add(row)
    db.commit()


def build_graph(db_factory: Callable[[], Session], llm: LLM, dropout: DropoutModel | None = None,
                seed: int | None = None):
    dropout = dropout or DropoutModel()

    def assess_risk(s: DayState) -> DayState:
        with db_factory() as db:
            c = CheckIn(**s["checkin"])
            service.record_checkin(db, s["user_id"], c)
            feats = service.risk_features(db, s["user_id"])
            p = dropout.predict(feats)
            d = service.dashboard(db, s["user_id"], s["goal_id"])
        return {"risk": p, "band": risk_band(p), "ahead": d["days_ahead_of_pace"]}

    def route(s: DayState) -> str:
        if s["band"] == "high":
            return "recovery"
        if s["ahead"] < -2:
            return "replan"
        return "normal"

    def _quests(s: DayState, limit: int | None = None, energy_override: int | None = None) -> list[dict]:
        with db_factory() as db:
            energy = energy_override if energy_override is not None else s["checkin"]["energy"]
            ts = service.pick_today(db, s["goal_id"], energy, limit)
            return [{"id": t.id, "title": t.title, "minutes": t.est_minutes, "kind": t.kind,
                     "difficulty": t.difficulty, "boss": bool(t.is_boss),
                     "xp_preview": re_.task_xp(t.est_minutes, t.difficulty, t.kind)} for t in ts]

    def recovery(s: DayState) -> DayState:
        # Comeback mode: ONE easy task, the smallest on-ramp.
        return {"quests": _quests(s, limit=1, energy_override=1)}

    def replan(s: DayState) -> DayState:
        with db_factory() as db:
            from .db import Task
            remaining = sum(t.est_minutes for t in db.scalars(
                select(Task).where(Task.goal_id == s["goal_id"], Task.status != "done")).all())
            goal = db.get(Goal, s["goal_id"])
            hours = goal.plan_json.get("weekly_hours", 5)
            new_dl = re_.next_deadline_if_behind(service._today(), remaining, hours)
        return {"quests": _quests(s), "replan_deadline": new_dl.isoformat()}

    def normal(s: DayState) -> DayState:
        return {"quests": _quests(s)}

    def coach_node(s: DayState) -> DayState:
        with db_factory() as db:
            bandit = load_bandit(db, s["user_id"], seed)
            allowed = allowed_arms_for_state(s["checkin"]["energy"], s["checkin"]["mood"], s["risk"])
            arm = bandit.select(allowed)
            user = db.get(User, s["user_id"])
            goal = db.get(Goal, s["goal_id"])
            dash = service.dashboard(db, s["user_id"], s["goal_id"])
            facts = {"name": user.name, "why": goal.why, "streak": dash["streak"], "xp": dash["xp"],
                     "level": dash["level"], "progress_pct": dash["progress_pct"],
                     "days_ahead_of_pace": dash["days_ahead_of_pace"],
                     "today_quests": [q["title"] for q in s["quests"]], "risk_band": s["band"],
                     "replan_deadline": s.get("replan_deadline")}
            persona = user.coach_style
        msg = coach.write_message(llm, arm, persona, facts)
        tips = coach.focus_tips(s["checkin"]["energy"], s["checkin"].get("distractions", 0))
        return {"arm": arm, "message": msg, "tips": tips}

    g = StateGraph(DayState)
    g.add_node("assess_risk", assess_risk)
    g.add_node("recovery", recovery)
    g.add_node("replan", replan)
    g.add_node("normal", normal)
    g.add_node("coach", coach_node)
    g.set_entry_point("assess_risk")
    g.add_conditional_edges("assess_risk", route, {"recovery": "recovery", "replan": "replan", "normal": "normal"})
    for n in ("recovery", "replan", "normal"):
        g.add_edge(n, "coach")
    g.add_edge("coach", END)
    return g.compile()


def record_outcome(db: Session, user_id: int, arm: str, completed_within_24h: bool, seed: int | None = None) -> None:
    """Close the bandit's feedback loop. Call from a scheduled job ~24h after each message."""
    b = load_bandit(db, user_id, seed)
    b.update(arm, int(completed_within_24h))
    b.decay()
    save_bandit(db, user_id, b)
