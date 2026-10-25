"""Thompson-sampling contextual-free bandit that learns which coaching style
(arm) actually gets *this* user to complete tasks. Reward = 1 if the user
completed >=1 task within 24h of the message, else 0."""
from __future__ import annotations

import numpy as np

ARMS = ["celebrate", "gentle_nudge", "challenge", "why_reminder", "tiny_step"]


class ThompsonBandit:
    def __init__(self, arms: list[str] = ARMS, seed: int | None = None):
        self.arms = arms
        self.alpha = {a: 1.0 for a in arms}
        self.beta = {a: 1.0 for a in arms}
        self.rng = np.random.default_rng(seed)

    def select(self, allowed: list[str] | None = None) -> str:
        pool = allowed or self.arms
        samples = {a: self.rng.beta(self.alpha[a], self.beta[a]) for a in pool}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: int) -> None:
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def decay(self, factor: float = 0.98) -> None:
        """Forget old evidence slowly: people's motivators drift over time."""
        for a in self.arms:
            self.alpha[a] = 1 + (self.alpha[a] - 1) * factor
            self.beta[a] = 1 + (self.beta[a] - 1) * factor

    def mean(self, arm: str) -> float:
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])


def allowed_arms_for_state(energy: int, mood: int, risk: float) -> list[str]:
    """Safety rails so exploration never picks a tone that is inappropriate.
    A burnt-out user is never sent the 'challenge' arm."""
    if energy <= 2 or mood <= 2 or risk >= 0.7:
        return ["gentle_nudge", "tiny_step", "why_reminder"]
    if energy >= 4 and mood >= 4:
        return ["celebrate", "challenge", "why_reminder"]
    return ARMS
