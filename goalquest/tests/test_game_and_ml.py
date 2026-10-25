import random
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.game import reward_engine as re_
from app.ml.bandit import ThompsonBandit, allowed_arms_for_state
from app.ml.dropout import DropoutModel, heuristic_risk, risk_band
from app.ml.estimator import DurationCalibrator
from app.ml.knowledge import BKT, SM2Card
from app.schemas import PlanSpec
from tests.conftest import make_plan


# ---- reward engine ----
def test_xp_scales_and_verification_weights():
    assert re_.task_xp(60, 3, "do") > re_.task_xp(60, 1, "do") > 0
    assert re_.task_xp(60, 2, "learn", 0.5) < re_.task_xp(60, 2, "learn", 1.0)
    assert re_.task_xp(60, 2, "learn", 0.0) == 0
    # diminishing returns: 4x minutes gives < 4x XP
    assert re_.task_xp(60, 2, "learn") < 4 * re_.task_xp(15, 2, "learn")


def test_daily_cap_hard_stop_and_soft_diminish():
    assert re_.apply_daily_cap(400, 50) == 0
    assert re_.apply_daily_cap(0, 50) == 50
    assert re_.apply_daily_cap(230, 50) < 50          # past soft cap -> reduced
    assert re_.apply_daily_cap(390, 50) <= 10         # never exceeds cap


def test_levels_monotonic():
    assert re_.level_for_xp(0) == 1
    levels = [re_.level_for_xp(x) for x in range(0, 5000, 50)]
    assert levels == sorted(levels) and levels[-1] > 5
    assert re_.rank_for_level(1) == "Apprentice" and re_.rank_for_level(12) == "Adept"


def test_streak_extend_freeze_reset():
    d0 = date(2026, 1, 1)
    s, e = re_.update_streak(re_.StreakState(), d0)
    assert (s.streak, e) == (1, "started")
    s, e = re_.update_streak(s, d0 + timedelta(days=1))
    assert (s.streak, e) == (2, "extended")
    s, e = re_.update_streak(s, d0 + timedelta(days=3))        # missed 1 day, has 1 freeze
    assert e == "freeze_used" and s.streak == 3 and s.freezes == 0
    s, e = re_.update_streak(s, d0 + timedelta(days=10))       # missed many, no freezes
    assert e == "reset" and s.streak == 1
    s2, e2 = re_.update_streak(s, s.last_active)
    assert e2 == "already_counted"


def test_streak_earns_freeze_every_7():
    s = re_.StreakState(freezes=0)
    d = date(2026, 1, 1)
    for i in range(7):
        s, _ = re_.update_streak(s, d + timedelta(days=i))
    assert s.streak == 7 and s.freezes == 1


def test_multiplier_capped():
    assert re_.streak_multiplier(0) == 1.0 and re_.streak_multiplier(1000) == 1.5


def test_badges_and_loot():
    ctx = re_.BadgeContext(tasks_done_total=1, streak=7, boss_cleared=True, days_away_before_return=4)
    got = set(re_.new_badges(ctx))
    assert {"first_blood", "week_warrior", "boss_slayer", "comeback_kid"} <= got
    ctx.existing = got
    assert re_.new_badges(ctx) == []
    assert re_.roll_loot(5, random.Random(1)) is None                 # too small to drop
    drops = [re_.roll_loot(20, random.Random(i)) for i in range(200)]
    rate = sum(d is not None for d in drops) / 200
    assert 0.2 < rate < 0.4


def test_pace_helpers():
    assert re_.days_ahead_of_pace(500, 1000, 10, 20) == 0
    assert re_.days_ahead_of_pace(300, 1000, 10, 20) < 0
    assert re_.next_deadline_if_behind(date(2026, 1, 1), 600, 7) > date(2026, 1, 1)


# ---- ML ----
def test_bandit_learns_best_arm():
    rng = random.Random(0)
    b = ThompsonBandit(seed=0)
    true_p = {"celebrate": 0.3, "gentle_nudge": 0.35, "challenge": 0.2, "why_reminder": 0.8, "tiny_step": 0.4}
    picks = []
    for _ in range(600):
        a = b.select()
        picks.append(a)
        b.update(a, int(rng.random() < true_p[a]))
    assert picks[-100:].count("why_reminder") > 60


def test_bandit_safety_rails():
    assert "challenge" not in allowed_arms_for_state(energy=1, mood=3, risk=0.1)
    assert "challenge" not in allowed_arms_for_state(energy=4, mood=4, risk=0.9)
    b = ThompsonBandit(seed=1)
    for _ in range(50):
        assert b.select(allowed_arms_for_state(1, 1, 0.9)) != "challenge"


def test_bkt_and_sm2():
    k = BKT()
    start = k.p_known
    for _ in range(6):
        k.update(True)
    assert k.p_known > 0.9 > start
    before = k.p_known
    k.update(False)
    assert k.p_known < before
    c = SM2Card()
    today = date(2026, 1, 1)
    ivs = [c.review(5, today).interval_days for _ in range(3)]
    assert ivs[0] == 1 and ivs[1] == 6 and ivs[2] > 6
    assert c.review(1, today).interval_days == 1 and c.reps == 0


def _feat(**kw):
    base = dict(days_since_active=0, completion_rate_7d=0.8, avg_energy_7d=4, avg_mood_7d=4,
                focus_minutes_trend=2, distraction_rate_7d=0.1, streak_broken_recently=0)
    base.update(kw)
    return base


def test_dropout_heuristic_orders_risk():
    healthy, struggling = heuristic_risk(_feat()), heuristic_risk(
        _feat(days_since_active=5, completion_rate_7d=0.0, avg_energy_7d=1.5, avg_mood_7d=2,
              focus_minutes_trend=-10, distraction_rate_7d=0.8, streak_broken_recently=1))
    assert healthy < 0.3 < 0.7 < struggling
    assert risk_band(struggling) == "high" and risk_band(healthy) == "low"


def test_dropout_model_fits_and_explains():
    rng = random.Random(0)
    rows, labels = [], []
    for _ in range(120):
        bad = rng.random() < 0.5
        rows.append(_feat(days_since_active=rng.randint(3, 7) if bad else rng.randint(0, 1),
                          completion_rate_7d=rng.uniform(0, .3) if bad else rng.uniform(.6, 1),
                          avg_energy_7d=rng.uniform(1, 2.5) if bad else rng.uniform(3, 5)))
        labels.append(int(bad))
    m = DropoutModel()
    assert m.fit(rows, labels)
    assert m.predict(rows[labels.index(1)]) > 0.5 > m.predict(rows[labels.index(0)])
    assert m.explain(rows[labels.index(1)])
    assert not DropoutModel().fit(rows[:10], labels[:10])   # too little data -> heuristic fallback


def test_duration_calibrator_learns_optimism_bias():
    c = DurationCalibrator()
    for _ in range(20):
        c.observe("learn", 30, 60)
    assert c.adjust("learn", 30) > 45
    assert c.adjust("do", 30) == 30


# ---- schema validation ----
def test_plan_rejects_cycles_unknown_deps_and_long_tasks():
    d = make_plan().model_dump(mode="json")
    d["chapters"][0]["quests"][0]["tasks"][0]["depends_on"] = ["t3"]    # t1->t3->t2->t1 cycle
    with pytest.raises(ValidationError, match="cycle"):
        PlanSpec.model_validate(d)
    d = make_plan().model_dump(mode="json")
    d["chapters"][0]["quests"][0]["tasks"][1]["depends_on"] = ["nope"]
    with pytest.raises(ValidationError, match="unknown"):
        PlanSpec.model_validate(d)
    d = make_plan().model_dump(mode="json")
    d["chapters"][0]["quests"][0]["tasks"][0]["minutes"] = 200
    with pytest.raises(ValidationError):
        PlanSpec.model_validate(d)
