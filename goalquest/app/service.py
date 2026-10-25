"""Service layer: all state changes go through here. Agents and tools never
touch the DB directly, so game rules are enforced in exactly one place."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import BadgeAward, CheckInRow, Goal, Task, User, XPLedger
from .game import reward_engine as re_
from .schemas import CheckIn, PlanSpec


def _today() -> date:
    return datetime.now(timezone.utc).date()


# ---------- goals & plans ----------

def save_plan(db: Session, user_id: int, plan: PlanSpec, today: date | None = None) -> Goal:
    today = today or _today()
    goal = Goal(
        user_id=user_id,
        title=plan.goal,
        why=plan.why,
        deadline=today + timedelta(days=plan.deadline_days),
        plan_json=plan.model_dump(mode="json"),
    )
    db.add(goal)
    db.flush()
    for ch in plan.chapters:
        for q in ch.quests:
            for t in q.tasks:
                db.add(Task(
                    goal_id=goal.id, key=t.key, title=t.title, kind=t.kind.value,
                    est_minutes=t.minutes, difficulty=int(t.difficulty), done_when=t.done_when,
                    depends_on=t.depends_on, is_boss=int(q.is_boss),
                ))
    db.commit()
    return goal


def unlocked_tasks(db: Session, goal_id: int) -> list[Task]:
    tasks = db.scalars(select(Task).where(Task.goal_id == goal_id)).all()
    done = {t.key for t in tasks if t.status == "done"}
    return [t for t in tasks if t.status != "done" and all(d in done for d in t.depends_on)]


def pick_today(db: Session, goal_id: int, energy: int, limit: int | None = None) -> list[Task]:
    """Energy-matched selection: low energy -> easy/review first; high -> hard/do first."""
    limit = limit or settings.max_tasks_per_day
    cands = unlocked_tasks(db, goal_id)
    if energy <= 2:
        cands.sort(key=lambda t: (t.difficulty, t.est_minutes))
    elif energy >= 4:
        cands.sort(key=lambda t: (-t.is_boss, -t.difficulty, t.id))
    else:
        cands.sort(key=lambda t: (t.difficulty, t.id))
    return cands[:limit]


# ---------- XP / completion ----------

def xp_today(db: Session, user_id: int, day: date | None = None) -> int:
    day = day or _today()
    return db.scalar(select(func.coalesce(func.sum(XPLedger.amount), 0))
                     .where(XPLedger.user_id == user_id, XPLedger.day == day)) or 0


def total_xp(db: Session, user_id: int) -> int:
    return db.scalar(select(func.coalesce(func.sum(XPLedger.amount), 0))
                     .where(XPLedger.user_id == user_id)) or 0


def complete_task(db: Session, user_id: int, task_id: int, verified_score: float,
                  today: date | None = None, rng=None) -> dict:
    today = today or _today()
    task = db.get(Task, task_id)
    user = db.get(User, user_id)
    if task is None or user is None:
        raise ValueError("unknown task or user")
    if task.status == "done":
        return {"ok": False, "reason": "already_completed"}
    if any(t.status != "done" for t in db.scalars(select(Task).where(
            Task.goal_id == task.goal_id, Task.key.in_(task.depends_on or []))).all()):
        return {"ok": False, "reason": "dependencies_not_met"}

    level_before = re_.level_for_xp(total_xp(db, user_id))

    # streak first (it feeds the multiplier)
    prev_last = user.last_active
    st, event = re_.update_streak(
        re_.StreakState(user.streak, user.freezes, user.last_active), today, settings.streak_freeze_every_n_days)
    user.streak, user.freezes, user.last_active = st.streak, st.freezes, st.last_active
    days_away = (today - prev_last).days - 1 if prev_last else 0

    raw = re_.task_xp(task.est_minutes, task.difficulty, task.kind, verified_score)
    raw = int(round(raw * re_.streak_multiplier(user.streak)))
    if task.is_boss:
        raw = int(raw * 1.5)
    xp = re_.apply_daily_cap(xp_today(db, user_id, today), raw, settings.daily_xp_cap)

    task.status, task.completed_at = "done", datetime.now(timezone.utc)
    db.add(XPLedger(user_id=user_id, amount=xp, reason=f"task:{task.key}", day=today))

    loot = re_.roll_loot(xp, rng)
    if loot:
        name, bonus = loot
        if bonus:
            db.add(XPLedger(user_id=user_id, amount=bonus, reason=f"loot:{name}", day=today))
        if name == "streak_freeze":
            user.freezes += 1

    # badges
    existing = set(db.scalars(select(BadgeAward.code).where(BadgeAward.user_id == user_id)).all())
    week_ago = today - timedelta(days=7)
    deep = db.scalar(select(func.count()).select_from(CheckInRow).where(
        CheckInRow.user_id == user_id, CheckInRow.day >= week_ago, CheckInRow.minutes_focused >= 50)) or 0
    done_total = db.scalar(select(func.count()).select_from(Task).where(
        Task.goal_id == task.goal_id, Task.status == "done")) or 0
    ctx = re_.BadgeContext(done_total, deep, max(days_away, 0), bool(task.is_boss), user.streak, existing)
    badges = re_.new_badges(ctx)
    for b in badges:
        db.add(BadgeAward(user_id=user_id, code=b))
    db.commit()

    level_after = re_.level_for_xp(total_xp(db, user_id))
    return {
        "ok": True, "xp_awarded": xp, "loot": loot[0] if loot else None, "streak": user.streak,
        "streak_event": event, "new_badges": badges, "leveled_up": level_after > level_before,
        "level": level_after, "rank": re_.rank_for_level(level_after), "total_xp": total_xp(db, user_id),
    }


# ---------- check-ins & features ----------

def record_checkin(db: Session, user_id: int, c: CheckIn, tasks_done: int = 0, day: date | None = None) -> CheckInRow:
    row = CheckInRow(user_id=user_id, day=day or _today(), energy=c.energy, mood=c.mood,
                     minutes_focused=c.minutes_focused, distractions=c.distractions,
                     tasks_done=tasks_done, note=c.note)
    db.add(row)
    db.commit()
    return row


def risk_features(db: Session, user_id: int, today: date | None = None) -> dict:
    today = today or _today()
    user = db.get(User, user_id)
    rows = db.scalars(select(CheckInRow).where(
        CheckInRow.user_id == user_id, CheckInRow.day >= today - timedelta(days=7))
        .order_by(CheckInRow.day)).all()
    n = len(rows)
    # A user who has never been active is new, not disengaged: 0 days, not 7.
    days_inactive = (today - user.last_active).days if user.last_active else 0
    if n == 0:
        return {"days_since_active": days_inactive,
                "completion_rate_7d": 0.0, "avg_energy_7d": 3.0, "avg_mood_7d": 3.0,
                "focus_minutes_trend": 0.0, "distraction_rate_7d": 0.0, "streak_broken_recently": 0}
    focus = [r.minutes_focused for r in rows]
    slope = 0.0
    if n >= 2:
        xs = list(range(n))
        mx, my = sum(xs) / n, sum(focus) / n
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, focus)) / den if den else 0.0
    return {
        "days_since_active": days_inactive,
        "completion_rate_7d": min(sum(r.tasks_done for r in rows) / max(n * settings.max_tasks_per_day, 1), 1.0),
        "avg_energy_7d": sum(r.energy for r in rows) / n,
        "avg_mood_7d": sum(r.mood for r in rows) / n,
        "focus_minutes_trend": slope,
        "distraction_rate_7d": min(sum(r.distractions for r in rows) / max(sum(focus), 1) * 25, 1.0),
        "streak_broken_recently": int(user.streak <= 1 and n > 2),
    }


def dashboard(db: Session, user_id: int, goal_id: int, today: date | None = None) -> dict:
    today = today or _today()
    goal = db.get(Goal, goal_id)
    user = db.get(User, user_id)
    if goal is None or user is None:
        raise ValueError("unknown goal or user")
    tasks = db.scalars(select(Task).where(Task.goal_id == goal_id)).all()
    total_min = sum(t.est_minutes for t in tasks)
    done_min = sum(t.est_minutes for t in tasks if t.status == "done")
    plan_days = goal.plan_json.get("deadline_days", 1)
    if goal.deadline:
        elapsed = plan_days - (goal.deadline - today).days
    else:
        elapsed = 0
    xp = total_xp(db, user_id)
    lv = re_.level_for_xp(xp)
    return {
        "user_name": user.name,
        "goal": goal.title, "progress_pct": round(100 * done_min / total_min, 1) if total_min else 0.0,
        "days_ahead_of_pace": round(re_.days_ahead_of_pace(done_min, total_min, elapsed, plan_days), 1),
        "xp": xp, "level": lv, "rank": re_.rank_for_level(lv),
        "xp_to_next": re_.xp_for_level(lv + 1) - xp, "streak": user.streak, "freezes": user.freezes,
        "today_xp": xp_today(db, user_id, today),
        "badges": sorted(db.scalars(select(BadgeAward.code).where(BadgeAward.user_id == user_id)).all()),
    }
