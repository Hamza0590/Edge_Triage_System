"""Condition-based bounded FIFO with explicit lifecycle and loss accounting."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from edge_triage.runtime.interfaces import CandidateInput


class QueueClosed(RuntimeError):
    pass


class QueueOverflow(RuntimeError):
    pass


class BoundedCandidateQueue:
    def __init__(
        self,
        capacity: int,
        overflow: str = "block",
        *,
        event: Callable[[str, CandidateInput], None] | None = None,
    ) -> None:
        if capacity < 1 or overflow not in ("block", "reject", "drop"):
            raise ValueError("invalid queue configuration")
        self.capacity, self.overflow, self.event = capacity, overflow, event
        self.condition = threading.Condition()
        self.items: deque[tuple[CandidateInput, float]] = deque()
        self.pending: set[str] = set()
        self.active: set[str] = set()
        self.closed = False
        self.counts = dict.fromkeys(
            (
                "enqueued",
                "dequeued",
                "completed",
                "failed",
                "rejected",
                "dropped",
                "cancelled",
                "peak_length",
                "blocked_puts",
            ),
            0,
        )

    def _event(self, name: str, item: CandidateInput) -> None:
        if self.event:
            self.event(name, item)

    def put(self, item: CandidateInput) -> None:
        key = item.candidate.candidate_id
        with self.condition:
            if key in self.pending or key in self.active:
                raise ValueError("duplicate live candidate")
            if self.closed:
                raise QueueClosed("queue closed")
            blocked = False
            while len(self.items) >= self.capacity and not self.closed:
                if self.overflow != "block":
                    reason = "rejected" if self.overflow == "reject" else "dropped"
                    self.counts[reason] += 1
                    self._event(reason, item)
                    raise QueueOverflow(reason)
                if not blocked:
                    self.counts["blocked_puts"] += 1
                    self._event("backpressure", item)
                    blocked = True
                self.condition.wait()
            if self.closed:
                raise QueueClosed("queue closed while enqueue blocked")
            self.items.append((item, time.monotonic()))
            self.pending.add(key)
            self.counts["enqueued"] += 1
            self.counts["peak_length"] = max(self.counts["peak_length"], len(self.items))
            self._event("enqueued", item)
            self.condition.notify_all()

    def get(self) -> CandidateInput:
        with self.condition:
            while not self.items and not self.closed:
                self.condition.wait()
            if not self.items:
                raise QueueClosed("queue drained")
            item, _ = self.items.popleft()
            key = item.candidate.candidate_id
            self.pending.remove(key)
            self.active.add(key)
            self.counts["dequeued"] += 1
            self._event("dequeued", item)
            self.condition.notify_all()
            return item

    def done(self, item: CandidateInput, failed: bool = False) -> None:
        with self.condition:
            self.active.remove(item.candidate.candidate_id)
            outcome = "failed" if failed else "completed"
            self.counts[outcome] += 1
            self._event(outcome, item)

    def close(self, cancel: bool = False) -> None:
        with self.condition:
            self.closed = True
            if cancel:
                while self.items:
                    item, _ = self.items.popleft()
                    self.pending.remove(item.candidate.candidate_id)
                    self.counts["cancelled"] += 1
                    self._event("cancelled", item)
            self.condition.notify_all()

    def stats(self) -> dict[str, Any]:
        with self.condition:
            return {
                **self.counts,
                "length": len(self.items),
                "active": len(self.active),
                "oldest_wait_ms": (time.monotonic() - self.items[0][1]) * 1000
                if self.items
                else 0.0,
                "capacity": self.capacity,
                "overflow": self.overflow,
                "closed": self.closed,
            }

    def __len__(self) -> int:
        with self.condition:
            return len(self.items)
