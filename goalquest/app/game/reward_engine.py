"""Deterministic reward engine. Deliberately contains NO LLM calls: rewards must
be consistent, testable and impossible to talk the model into inflating."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

# ---------- XP ----------

KIND_MULT = {"learn": 1.0, "do": 1.3, "review": 0.8}


def task_xp(est_minutes: int, difficulty: int, kind: str, verified_score: float = 1.0) -> int:
    """XP scales with effort and difficulty, and is weighted by verification score.

    Using sqrt(minutes) gives diminishing returns, so padding a task's duration
    does not pay off linearly.
    """
    base = 10 * math.sqrt(max(est_minutes, 1) / 15.0)
    xp = base * (1 + 0.5 * (difficulty - 1)) * KIND_MULT.get(kind, 1.0) * max(0.0, min(verified_score, 1.0))
    return int(round(xp))


def apply_daily_cap(awarded_today: int, new_xp: int, cap: int = 400) -> int:
    """Diminishing returns past 60% of the cap, hard stop at the cap."""
    soft = int(cap * 0.6)
    remaining = max(cap - awarded_today, 0)
    if remaining == 0:
        return 0
    over_soft = max(awarded_today + new_xp - soft, 0)
    if over_soft > 0:
        reducible = min(over_soft, new_xp)
        new_xp = new_xp - reducible + int(reducible * 0.5)
    return min(new_xp, remaining)


# ---------- Levels ----------

def xp_for_level(level: int) -> int:
    """Cumulative XP needed to reach `level` (level 1 = 0 XP). Gentle quadratic curve."""
    return int(50 * (level - 1) ** 1.6)


def level_for_xp(total_xp: int) -> int:
    level = 1
    while xp_for_level(level + 1) <= total_xp:
        level += 1
    return level


RANKS = [(1, "Apprentice"), (5, "Practitioner"), (10, "Adept"), (20, "Expert"), (35, "Master")]


def rank_for_level(level: int) -> str:
    name = RANKS[0][1]
    for lv, n in RANKS:
        if level >= lv:
            name = n
    return name


# ---------- Streaks with freeze tokens ----------

@dataclass
class StreakState:
    streak: int = 0
    freezes: int = 1
    last_active: date | None = None


def update_streak(state: StreakState, today: date, freeze_every: int = 7) -> tuple[StreakState, str]:
    """Advance the streak for activity on `today`. Missing exactly the days in
    between consumes one freeze token each; if tokens run out the streak resets
    (to 1, since today counts). Returns (new_state, event)."""
    if state.last_active == today:
        return state, "already_counted"
    if state.last_active is None:
        new = StreakState(1, state.freezes, today)
        return new, "started"
    gap = (today - state.last_active).days
    if gap < 1:
        return state, "already_counted"
    missed = gap - 1
    freezes = state.freezes
    if missed == 0:
        streak, event = state.streak + 1, "extended"
    elif missed <= freezes:
        freezes -= missed
        streak, event = state.streak + 1, "freeze_used"
    else:
        streak, event = 1, "reset"
        freezes = max(freezes, 0)
    if streak > 0 and streak % freeze_every == 0 and event != "reset":
        freezes += 1  # earn a token every N days
    return StreakState(streak, freezes, today), event


def streak_multiplier(streak: int) -> float:
    """Capped multiplier: +2% per streak day, max 1.5x."""
    return min(1.0 + 0.02 * streak, 1.5)


# ---------- Badges ----------

@dataclass
class BadgeContext:
    tasks_done_total: int = 0
    focus_sessions_50plus_this_week: int = 0
    days_away_before_return: int = 0
    boss_cleared: bool = False
    streak: int = 0
    existing: set[str] = field(default_factory=set)


BADGE_RULES = {
    "first_blood": lambda c: c.tasks_done_total >= 1,
    "deep_work": lambda c: c.focus_sessions_50plus_this_week >= 3,
    "comeback_kid": lambda c: c.days_away_before_return >= 3,
    "boss_slayer": lambda c: c.boss_cleared,
    "week_warrior": lambda c: c.streak >= 7,
    "month_monk": lambda c: c.streak >= 30,
}


def new_badges(ctx: BadgeContext) -> list[str]:
    return [code for code, rule in BADGE_RULES.items() if code not in ctx.existing and rule(ctx)]


# ---------- Variable-ratio loot (earned-only, seeded for testability) ----------

LOOT_TABLE = [
    ("bonus_xp_small", 0.60, 10),
    ("bonus_xp_medium", 0.25, 25),
    ("streak_freeze", 0.10, 0),
    ("treat_token", 0.05, 0),
]


def roll_loot(xp_awarded: int, rng: random.Random | None = None) -> tuple[str, int] | None:
    """Loot only drops for meaningful completions (xp >= 8), ~30% of the time."""
    rng = rng or random.Random()
    if xp_awarded < 8 or rng.random() > 0.30:
        return None
    r, acc = rng.random(), 0.0
    for name, p, bonus in LOOT_TABLE:
        acc += p
        if r <= acc:
            return name, bonus
    return LOOT_TABLE[0][0], LOOT_TABLE[0][2]


# ---------- Re-pacing helper ----------

def days_ahead_of_pace(done_minutes: int, total_minutes: int, days_elapsed: int, deadline_days: int) -> float:
    """Positive = ahead of linear pace, negative = behind (in days)."""
    if total_minutes <= 0 or deadline_days <= 0:
        return 0.0
    expected_frac = days_elapsed / deadline_days
    actual_frac = done_minutes / total_minutes
    return (actual_frac - expected_frac) * deadline_days


def next_deadline_if_behind(today: date, remaining_minutes: int, weekly_hours: float, buffer: float = 0.2) -> date:
    per_day = (weekly_hours * 60) / 7
    days = math.ceil(remaining_minutes * (1 + buffer) / max(per_day, 1))
    return today + timedelta(days=days)
