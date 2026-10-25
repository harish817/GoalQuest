import random
from datetime import date, timedelta

from sqlalchemy import select

from app import service
from app.agents import planner, verifier
from app.db import BanditArm, Task, XPLedger
from app.graph import build_graph, record_outcome
from app.schemas import CheckIn
from tests.conftest import make_plan

TODAY = date(2026, 3, 1)


def _setup(db_factory, user_id):
    with db_factory() as db:
        return service.save_plan(db, user_id, make_plan(), today=TODAY).id


def test_dependency_unlocking_and_energy_matching(db_factory, user_id):
    gid = _setup(db_factory, user_id)
    with db_factory() as db:
        assert [t.key for t in service.unlocked_tasks(db, gid)] == ["t1"]      # only root unlocked
        tid = service.unlocked_tasks(db, gid)[0].id
        r = service.complete_task(db, user_id, tid, 1.0, today=TODAY, rng=random.Random(5))
        assert r["ok"] and r["xp_awarded"] > 0 and r["streak"] == 1 and "first_blood" in r["new_badges"]
        assert [t.key for t in service.unlocked_tasks(db, gid)] == ["t2"]
        # completing twice is rejected; locked task is rejected
        assert service.complete_task(db, user_id, tid, 1.0, today=TODAY)["reason"] == "already_completed"
        locked = db.scalars(select(Task).where(Task.key == "t4")).one()
        assert service.complete_task(db, user_id, locked.id, 1.0, today=TODAY)["reason"] == "dependencies_not_met"


def test_xp_ledger_is_source_of_truth_and_daily_cap(db_factory, user_id):
    gid = _setup(db_factory, user_id)
    with db_factory() as db:
        r = service.complete_task(db, user_id, service.unlocked_tasks(db, gid)[0].id, 1.0, today=TODAY,
                                  rng=random.Random(1))
        ledger_sum = sum(x.amount for x in db.scalars(select(XPLedger)).all())
        assert ledger_sum == r["total_xp"] == service.total_xp(db, user_id)
        assert service.xp_today(db, user_id, TODAY) <= 400


def test_zero_verification_awards_no_xp(db_factory, user_id):
    gid = _setup(db_factory, user_id)
    with db_factory() as db:
        r = service.complete_task(db, user_id, service.unlocked_tasks(db, gid)[0].id, 0.0, today=TODAY)
        assert r["xp_awarded"] == 0


def test_boss_bonus_and_badge(db_factory, user_id):
    gid = _setup(db_factory, user_id)
    with db_factory() as db:
        for _ in range(4):
            t = service.unlocked_tasks(db, gid)[0]
            r = service.complete_task(db, user_id, t.id, 1.0, today=TODAY, rng=random.Random(9))
        assert t.key == "t4" and "boss_slayer" in r["new_badges"]
        d = service.dashboard(db, user_id, gid, today=TODAY + timedelta(days=5))
        assert d["progress_pct"] == 100.0 and d["days_ahead_of_pace"] > 0


def test_planner_retries_when_over_budget(fake_llm):
    # fake plan = 215 min; with 1 hour/week over 28 days budget is ~192 min -> retries then best-effort
    plan = planner.generate_plan(fake_llm, "g", "w", 28, 1.0)
    assert planner.total_minutes(plan) == 215
    assert sum(1 for c in fake_llm.calls if c[0] == "structured") == 3
    ok = planner.generate_plan(fake_llm, "g", "w", 28, 6.0)
    assert planner.total_minutes(ok) <= planner.budget_minutes(6.0, 28)


def test_verifier_updates_skill_state(fake_llm):
    from app.ml.knowledge import BKT
    q = verifier.make_quiz(fake_llm, "Vector spaces", "explain")
    k = BKT()
    before = k.p_known
    res = verifier.grade(fake_llm, q, "a basis spans the space", k)
    assert res.passed and k.p_known > before


def test_graph_routes_normal_and_recovery(db_factory, user_id, fake_llm):
    gid = _setup(db_factory, user_id)
    graph = build_graph(db_factory, fake_llm, seed=0)
    out = graph.invoke({"user_id": user_id, "goal_id": gid,
                        "checkin": CheckIn(energy=4, mood=4, minutes_focused=60).model_dump()})
    assert out["band"] in ("low", "medium") and out["quests"] and out["message"]
    assert out["arm"] in {"celebrate", "challenge", "why_reminder"}        # high energy+mood arms only
    assert out["tips"]

    # a disengaged user: many bad check-ins in the past, nothing completed -> recovery mode
    with db_factory() as db:
        for i in range(1, 6):
            service.record_checkin(db, user_id, CheckIn(energy=1, mood=1, distractions=5, minutes_focused=5),
                                   day=date.today() - timedelta(days=i))
    out = graph.invoke({"user_id": user_id, "goal_id": gid,
                        "checkin": CheckIn(energy=1, mood=1, distractions=4).model_dump()})
    assert out["band"] == "high"
    assert len(out["quests"]) == 1                                          # single easy on-ramp
    assert out["arm"] in {"gentle_nudge", "tiny_step", "why_reminder"}      # never 'challenge'


def test_bandit_outcome_persistence(db_factory, user_id):
    with db_factory() as db:
        record_outcome(db, user_id, "tiny_step", True, seed=0)
        record_outcome(db, user_id, "tiny_step", True, seed=0)
        rows = {r.arm: r for r in db.scalars(select(BanditArm).where(BanditArm.user_id == user_id)).all()}
        assert rows["tiny_step"].alpha > 1.5 and rows["challenge"].alpha == 1.0
