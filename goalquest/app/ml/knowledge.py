"""Knowledge modelling for the Verifier:
 - Bayesian Knowledge Tracing (BKT): probability the user has mastered a skill.
 - SM-2 spaced repetition: when to schedule the next review task."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class BKT:
    p_known: float = 0.2   # P(L0): prior probability skill already known
    p_learn: float = 0.15  # P(T): chance of learning on each opportunity
    p_slip: float = 0.10   # P(S): knows it but answers wrong
    p_guess: float = 0.20  # P(G): doesn't know but answers right

    def update(self, correct: bool) -> float:
        pk = self.p_known
        if correct:
            num = pk * (1 - self.p_slip)
            den = num + (1 - pk) * self.p_guess
        else:
            num = pk * self.p_slip
            den = num + (1 - pk) * (1 - self.p_guess)
        posterior = num / den if den else pk
        self.p_known = posterior + (1 - posterior) * self.p_learn
        return self.p_known

    @property
    def mastered(self) -> bool:
        return self.p_known >= 0.95


@dataclass
class SM2Card:
    ease: float = 2.5
    interval_days: int = 0
    reps: int = 0
    due: date | None = None

    def review(self, quality: int, today: date) -> "SM2Card":
        """quality: 0 (blackout) .. 5 (perfect)."""
        if quality < 3:
            self.reps = 0
            self.interval_days = 1
        else:
            if self.reps == 0:
                self.interval_days = 1
            elif self.reps == 1:
                self.interval_days = 6
            else:
                self.interval_days = round(self.interval_days * self.ease)
            self.reps += 1
        self.ease = max(1.3, self.ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        self.due = today + timedelta(days=self.interval_days)
        return self
