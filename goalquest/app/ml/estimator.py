"""Personal task-duration calibration. LLMs are optimistic planners, so we learn
a per-user (and per-kind) multiplier from actual-vs-estimated minutes using an
exponentially weighted average of the log-ratio (robust to outliers)."""
from __future__ import annotations

import math


class DurationCalibrator:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.log_ratio: dict[str, float] = {}

    def observe(self, kind: str, estimated: float, actual: float) -> None:
        if estimated <= 0 or actual <= 0:
            return
        r = math.log(actual / estimated)
        r = max(-1.0, min(r, 1.5))  # clip outliers
        prev = self.log_ratio.get(kind, 0.0)
        self.log_ratio[kind] = (1 - self.alpha) * prev + self.alpha * r

    def adjust(self, kind: str, estimated: float) -> int:
        return max(10, round(estimated * math.exp(self.log_ratio.get(kind, 0.0))))
