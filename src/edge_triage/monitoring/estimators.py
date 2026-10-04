"""Bounded time-window estimators and thread-safe execution counters."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

from edge_triage.contracts import ModelTier, TierJobs, TierLatency


class RollingWindow:
    def __init__(self, seconds: float, limit: int = 10000) -> None:
        if seconds <= 0 or limit < 2:
            raise ValueError("positive window and capacity >=2 required")
        self.seconds, self.limit = seconds, limit
        self.values: deque[tuple[float, float]] = deque()
        self.truncated_until = -1.0
        self.last_time = -1.0

    def prune(self, now: float) -> None:
        while self.values and self.values[0][0] < now - self.seconds:
            self.values.popleft()

    def add(self, value: float, now: float) -> None:
        if not math.isfinite(now) or not math.isfinite(value) or now < self.last_time or value < 0:
            raise ValueError("monotonic timestamps and nonnegative values required")
        self.last_time = now
        self.prune(now)
        if len(self.values) == self.limit:
            timestamp, _ = self.values.popleft()
            self.truncated_until = timestamp + self.seconds
        self.values.append((now, value))

    def summary(self, now: float) -> dict[str, Any]:
        self.prune(now)
        n = len(self.values)
        truncated = now <= self.truncated_until
        span = self.values[-1][0] - self.values[0][0] if n > 1 else 0
        return {
            "samples": n,
            "mean": sum(v for _, v in self.values) / n if n and not truncated else None,
            "rate": (n - 1) / span if span > 0 and not truncated else None,
            "truncated": truncated,
        }


class RuntimeCounters:
    def __init__(self, window_s: float = 5, limit: int = 10000) -> None:
        self.lock = threading.RLock()
        self.window_s = window_s
        self.arrivals = RollingWindow(window_s, limit)
        self.latencies = {tier: RollingWindow(window_s, limit) for tier in ModelTier}
        self.active = dict.fromkeys(ModelTier, 0)

    def arrival(self, now: float | None = None) -> None:
        with self.lock:
            self.arrivals.add(1, time.monotonic() if now is None else now)

    def started(self, tier: ModelTier) -> None:
        with self.lock:
            self.active[tier] += 1

    def finished(
        self, tier: ModelTier, duration_ms: float | None, now: float | None = None
    ) -> None:
        with self.lock:
            if self.active[tier] < 1:
                raise ValueError("active job underflow")
            self.active[tier] -= 1
            if duration_ms is not None:
                self.latencies[tier].add(duration_ms, time.monotonic() if now is None else now)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        with self.lock:
            timestamp = time.monotonic() if now is None else now
            arrival = self.arrivals.summary(timestamp)
            latency = {t: w.summary(timestamp) for t, w in self.latencies.items()}
            n = sum(v["samples"] for v in latency.values())
            usable = all(v["mean"] is not None or v["samples"] == 0 for v in latency.values())
            return {
                "arrival_rate_per_s": arrival["rate"],
                "arrival_samples": arrival["samples"],
                "active_jobs_total": sum(self.active.values()),
                "active_jobs_by_tier": tuple(
                    TierJobs(tier=t, count=n) for t, n in self.active.items()
                ),
                "recent_latency_by_tier": tuple(
                    TierLatency(tier=t, mean_ms=v["mean"], samples=v["samples"])
                    for t, v in latency.items()
                ),
                "latency_samples": n,
                "recent_latency_ms": sum((v["mean"] or 0) * v["samples"] for v in latency.values())
                / n
                if n and usable
                else None,
                "estimator_window_s": self.window_s,
            }
