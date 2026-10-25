"""Churn / drop-off risk model.

Features are computed from recent check-ins. Until enough labelled history
exists, a transparent heuristic prior is used; once >= MIN_ROWS labelled
examples are available, a scikit-learn logistic regression takes over
(it stays interpretable: we can show the user *why* the risk is high).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "days_since_active",
    "completion_rate_7d",
    "avg_energy_7d",
    "avg_mood_7d",
    "focus_minutes_trend",   # slope of focus minutes over last 7 days (min/day)
    "distraction_rate_7d",
    "streak_broken_recently",
]
MIN_ROWS = 40


def heuristic_risk(f: dict) -> float:
    z = (
        -1.0
        + 0.45 * min(f["days_since_active"], 7)
        - 2.0 * f["completion_rate_7d"]
        - 0.35 * (f["avg_energy_7d"] - 3)
        - 0.35 * (f["avg_mood_7d"] - 3)
        - 0.02 * f["focus_minutes_trend"]
        + 0.8 * f["distraction_rate_7d"]
        + 0.9 * f["streak_broken_recently"]
    )
    return float(1 / (1 + np.exp(-z)))


class DropoutModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500)
        self.fitted = False

    @staticmethod
    def _vec(f: dict) -> list[float]:
        return [float(f[k]) for k in FEATURES]

    def fit(self, rows: list[dict], labels: list[int]) -> bool:
        """label = 1 if the user went inactive for >= 3 days after this snapshot."""
        if len(rows) < MIN_ROWS or len(set(labels)) < 2:
            return False
        X = self.scaler.fit_transform(np.array([self._vec(r) for r in rows]))
        self.clf.fit(X, labels)
        self.fitted = True
        return True

    def predict(self, f: dict) -> float:
        if not self.fitted:
            return heuristic_risk(f)
        X = self.scaler.transform(np.array([self._vec(f)]))
        return float(self.clf.predict_proba(X)[0, 1])

    def explain(self, f: dict, top: int = 3) -> list[tuple[str, float]]:
        """Return the features pushing risk up the most (fitted model only)."""
        if not self.fitted:
            return []
        x = self.scaler.transform(np.array([self._vec(f)]))[0]
        contrib = x * self.clf.coef_[0]
        order = np.argsort(-contrib)[:top]
        return [(FEATURES[i], float(contrib[i])) for i in order if contrib[i] > 0]


def risk_band(p: float) -> str:
    return "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"
