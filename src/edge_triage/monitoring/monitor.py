"""Background observation with bounded history, persisted samples and explicit costs."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from edge_triage.contracts import ResourceSnapshot, utc_now
from edge_triage.monitoring.collectors import Collector
from edge_triage.monitoring.estimators import RuntimeCounters


class ResourceMonitor:
    def __init__(
        self,
        collectors: list[Collector],
        counters: RuntimeCounters,
        queue_state: Callable[[], dict[str, Any]],
        *,
        interval_s: float = 0.5,
        history_limit: int = 256,
        path: Path | None = None,
        network: dict[str, Any] | None = None,
        warning_fraction: float = 0.1,
    ) -> None:
        if interval_s <= 0 or history_limit < 1:
            raise ValueError("invalid monitor interval/history")
        self.collectors, self.counters, self.queue_state = collectors, counters, queue_state
        self.interval_s, self.path = interval_s, path
        self.network, self.warning_fraction = network or {}, warning_fraction
        self.history: deque[ResourceSnapshot] = deque(maxlen=history_limit)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.sample_count = 0
        self.sampling_seconds = 0.0
        self.failures: list[str] = []
        self.latest: ResourceSnapshot | None = None
        self.started_at = 0.0
        self.elapsed_s = 0.0
        self.on_sample: Callable[[], None] | None = None

    def sample(self) -> ResourceSnapshot:
        values: dict[str, Any] = {"cpu_percent": None, "available_ram_bytes": None}
        availability = {}
        for collector in self.collectors:
            try:
                result = collector.collect()
                availability.update(result.pop("availability", {}))
                values.update(result)
            except Exception as error:
                availability[type(collector).__name__] = f"unavailable:{type(error).__name__}"
        values.update(self.counters.snapshot())
        state = self.queue_state()
        values.update(queue_length=state["length"], oldest_wait_ms=state["oldest_wait_ms"])
        values.update(self.network)
        return ResourceSnapshot(
            timestamp_utc=utc_now(),
            monotonic_ns=time.monotonic_ns(),
            snapshot_id=f"snapshot-{self.sample_count}",
            availability=availability,
            **values,
        )

    def _loop(self) -> None:
        stream = None
        try:
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                stream = self.path.open("a", encoding="utf-8", newline="\n")
            while not self.stop_event.is_set():
                begin = time.perf_counter()
                snapshot = self.sample()
                if stream:
                    stream.write(snapshot.model_dump_json() + "\n")
                    stream.flush()
                with self.lock:
                    self.latest = snapshot
                    self.history.append(snapshot)
                    self.sample_count += 1
                    self.sampling_seconds += time.perf_counter() - begin
                if self.on_sample is not None:
                    self.on_sample()
                self.stop_event.wait(max(0, self.interval_s - (time.perf_counter() - begin)))
        except Exception as error:
            with self.lock:
                self.failures.append(f"{type(error).__name__}: {error}")
        finally:
            if stream:
                stream.close()

    def start(self) -> None:
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.started_at = time.perf_counter()
            self.thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            thread = self.thread
            self.stop_event.set()
        if thread:
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("monitor collector did not stop")
            with self.lock:
                if self.thread is not None:
                    self.elapsed_s += time.perf_counter() - self.started_at
                    self.thread = None

    def current(self, queue_length: int = 0) -> ResourceSnapshot:
        with self.lock:
            if self.failures:
                raise RuntimeError("monitor failed: " + self.failures[-1])
            if self.latest is not None:
                return self.latest
        # Startup snapshot is explicitly unavailable; no collector blocks inference.
        return ResourceSnapshot(
            timestamp_utc=utc_now(),
            monotonic_ns=time.monotonic_ns(),
            cpu_percent=None,
            available_ram_bytes=None,
            queue_length=queue_length,
            arrival_rate_per_s=None,
            availability={"monitor": "warming_up"},
        )

    def overhead(self) -> dict[str, Any]:
        with self.lock:
            elapsed = self.elapsed_s + (time.perf_counter() - self.started_at if self.thread else 0)
            fraction = self.sampling_seconds / elapsed if elapsed else 0
            return {
                "samples": self.sample_count,
                "sampling_seconds": self.sampling_seconds,
                "elapsed_s": elapsed,
                "sampling_wall_fraction": fraction,
                "warning": fraction > self.warning_fraction,
                "warning_threshold": self.warning_fraction,
                "failures": self.failures.copy(),
                "interval_s": self.interval_s,
                "scope": "collector+serialization+flush wall time; not process CPU attribution",
            }
