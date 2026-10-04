"""Monitor-only sequential consumer driven by a bounded asynchronous arrival producer."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from edge_triage.config import AppConfig
from edge_triage.contracts import (
    AdmissionDecision,
    Candidate,
    ModelTier,
    PredictionResult,
    ResourceSnapshot,
    SchedulingDecision,
    UncertaintyResult,
    utc_now,
)
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.data.loaders import MeerlichtDataset
from edge_triage.experiments.workloads import TraceRecord, load_trace
from edge_triage.health import hash_file
from edge_triage.models.registry import read_index
from edge_triage.models.training import resolve_training_data
from edge_triage.monitoring.collectors import HostCollector, NvidiaCollector
from edge_triage.monitoring.estimators import RuntimeCounters
from edge_triage.monitoring.monitor import ResourceMonitor
from edge_triage.runs import RunManager
from edge_triage.runtime.bounded_queue import BoundedCandidateQueue, QueueClosed, QueueOverflow
from edge_triage.runtime.components import LoggerEventSink
from edge_triage.runtime.concurrent import consume_jobs, create_executor
from edge_triage.runtime.executors import FixedConcurrencyExecutor
from edge_triage.runtime.factory import assemble
from edge_triage.runtime.interfaces import (
    AdmissionPolicy,
    CandidateInput,
    ImageUncertaintyPolicy,
    InferenceExecutor,
    ModelAdapter,
    TierAdmissionPolicy,
    TierSelectionPolicy,
    UncertaintyPolicy,
)
from edge_triage.runtime.pipeline import Pipeline, QueueEntry
from edge_triage.runtime.records import ParquetRecordWriter
from edge_triage.runtime.state import PipelineState, StateMachine


class ObservedExecutor:
    def __init__(self, base: InferenceExecutor, counters: RuntimeCounters) -> None:
        self.base, self.counters = base, counters

    def execute(
        self, model: ModelAdapter, item: CandidateInput, tier: ModelTier
    ) -> PredictionResult:
        self.counters.started(tier)
        start = time.perf_counter_ns()
        try:
            return self.base.execute(model, item, tier)
        finally:
            self.counters.finished(tier, (time.perf_counter_ns() - start) / 1e6)

    def close(self) -> None:
        self.base.close()


class ObservedUncertainty:
    """Keep active-job counters accurate without mixing UQ with profile latency."""

    def __init__(self, base: UncertaintyPolicy, counters: RuntimeCounters) -> None:
        self.base, self.counters = base, counters

    def evaluate(self, prediction: PredictionResult) -> UncertaintyResult:
        return self.base.evaluate(prediction)

    def evaluate_image(
        self, prediction: PredictionResult, image: torch.Tensor
    ) -> UncertaintyResult:
        self.counters.started(prediction.tier)
        try:
            if isinstance(self.base, ImageUncertaintyPolicy):
                return self.base.evaluate_image(prediction, image)
            return self.base.evaluate(prediction)
        finally:
            self.counters.finished(prediction.tier, None)


class ObservedSelector:
    def __init__(self, base: TierSelectionPolicy) -> None:
        self.base, self.total_ms, self.calls = base, 0.0, 0

    def select(self, candidate: Candidate, resources: ResourceSnapshot) -> SchedulingDecision:
        started = time.perf_counter_ns()
        try:
            return self.base.select(candidate, resources)
        finally:
            self.total_ms += (time.perf_counter_ns() - started) / 1e6
            self.calls += 1


class ObservedAdmission:
    def __init__(self, base: AdmissionPolicy, counters: RuntimeCounters) -> None:
        self.base, self.total_ms, self.calls = base, 0.0, 0
        self.counters = counters

    def decide(self, candidate: Candidate, resources: ResourceSnapshot) -> AdmissionDecision:
        return self._decide(candidate, None, resources)

    def admit(
        self, candidate: Candidate, selected_tier: ModelTier, resources: ResourceSnapshot
    ) -> AdmissionDecision:
        return self._decide(candidate, selected_tier, resources)

    def release(self, reservation_id: str, *, outcome: str = "completed") -> None:
        if isinstance(self.base, TierAdmissionPolicy):
            self.base.release(reservation_id, outcome=outcome)

    def _decide(
        self, candidate: Candidate, selected_tier: ModelTier | None, resources: ResourceSnapshot
    ) -> AdmissionDecision:
        started = time.perf_counter_ns()
        try:
            live = self.counters.snapshot()
            current = resources.model_copy(
                update={
                    "active_jobs_by_tier": live["active_jobs_by_tier"],
                    "active_jobs_total": live["active_jobs_total"],
                }
            )
            if isinstance(self.base, TierAdmissionPolicy) and selected_tier is not None:
                return self.base.admit(candidate, selected_tier, current)
            decision = self.base.decide(candidate, current)
            return decision.model_copy(
                update={
                    "reason_code": decision.reason_code + "_LIVE_COUNTERS",
                }
            )
        finally:
            self.total_ms += (time.perf_counter_ns() - started) / 1e6
            self.calls += 1


class TraceSource:
    def __init__(
        self, config: AppConfig, run_id: str, records: list[TraceRecord], stop: threading.Event
    ) -> None:
        self.dataset = MeerlichtDataset(config, "validation", run_id=run_id)
        self.positions = {row["index_no"]: i for i, row in enumerate(self.dataset.rows)}
        for row in records:
            if row.dataset_index not in self.positions:
                raise ValueError("trace attempts access outside validation")
            actual = self.dataset.rows[self.positions[row.dataset_index]]
            if actual["candidate_id"] != row.source_candidate_id:
                raise ValueError("trace source identity mismatch")
        self.run_id, self.records, self.stop = run_id, records, stop
        self.first_arrival: float | None = None
        self.last_arrival: float | None = None
        self.arrivals = 0
        self.max_lateness_ms = 0.0

    def __iter__(self) -> Iterator[CandidateInput]:
        origin = time.monotonic()
        for row in self.records:
            due = origin + row.relative_arrival_seconds
            if self.stop.wait(max(0, due - time.monotonic())):
                return
            sample = self.dataset[self.positions[row.dataset_index]]
            now = time.monotonic()
            self.first_arrival = now if self.first_arrival is None else self.first_arrival
            self.last_arrival = now
            self.arrivals += 1
            self.max_lateness_ms = max(self.max_lateness_ms, (now - due) * 1000)
            candidate = Candidate(
                run_id=self.run_id,
                trace_id=f"{self.run_id}:{row.candidate_id}",
                candidate_id=row.candidate_id,
                dataset_index=row.dataset_index,
                arrival_monotonic_ns=time.monotonic_ns(),
            )
            yield CandidateInput(candidate, sample[1], sample[2], 120000)

    def summary(self) -> dict[str, Any]:
        span = (
            (self.last_arrival - self.first_arrival)
            if self.last_arrival is not None and self.first_arrival is not None
            else 0
        )
        return {
            "offered": self.arrivals,
            "realized_arrival_rate_per_s": (self.arrivals - 1) / span if span > 0 else None,
            "max_arrival_lateness_ms": self.max_lateness_ms,
            "planned_duration_s": self.records[-1].relative_arrival_seconds,
            "repeated_occurrences": sum(r.repetition > 0 for r in self.records),
            "classification_observations": False,
            "backpressure_affects_realized_arrivals": True,
        }


def consume_stream(
    pipeline: Pipeline,
    source: TraceSource,
    queue: BoundedCandidateQueue,
    counters: RuntimeCounters,
    stop: threading.Event,
    executor: FixedConcurrencyExecutor | None = None,
) -> tuple[int, int]:
    entries: dict[str, QueueEntry] = {}
    lock = threading.Lock()
    errors: list[BaseException] = []

    def produce() -> None:
        try:
            for item in source:
                candidate = item.candidate
                counters.arrival()
                machine = StateMachine(
                    candidate, pipeline.logger.config_hash, LoggerEventSink(pipeline.logger)
                )
                entry = QueueEntry(machine, utc_now().isoformat(), time.monotonic_ns())
                pipeline._event(candidate, "data.arrived", "STREAM_ARRIVAL")
                machine.transition(PipelineState.QUEUED)
                with lock:
                    entries[candidate.candidate_id] = entry
                try:
                    queue.put(item)
                except QueueOverflow:
                    with lock:
                        entries.pop(candidate.candidate_id)
                    machine.transition(PipelineState.FAILED)
        except QueueClosed:
            if not stop.is_set():
                errors.append(RuntimeError("unexpected queue closure"))
        except BaseException as error:
            errors.append(error)
        finally:
            queue.close()

    producer = threading.Thread(target=produce, name="trace-producer", daemon=True)
    producer.start()
    completed = failed = 0
    try:
        if executor is not None:

            def get_entry() -> Any:
                try:
                    item = queue.get()
                except QueueClosed:
                    return None
                with lock:
                    return item, entries.pop(item.candidate.candidate_id)

            def finish(item: CandidateInput, record: Any) -> None:
                pipeline.writer.append(record)
                pipeline._event(
                    item.candidate, "inference.candidate_final", "CANDIDATE_FINAL", record
                )
                queue.done(item, record.status == "failed")

            async def process(
                item: CandidateInput, entry: QueueEntry, engine: FixedConcurrencyExecutor
            ) -> Any:
                return await pipeline.process_async(item, entry, engine, counters)

            completed, failed = asyncio.run(
                consume_jobs(
                    get_entry,
                    process,
                    finish,
                    executor,
                    pipeline.config.queue.capacity + executor.concurrency,
                )
            )
        else:
            while True:
                try:
                    item = queue.get()
                except QueueClosed:
                    break
                with lock:
                    entry = entries.pop(item.candidate.candidate_id)
                record = pipeline._process(item, entry)
                pipeline.writer.append(record)
                pipeline._event(
                    item.candidate, "inference.candidate_final", "CANDIDATE_FINAL", record
                )
                queue.done(item, record.status == "failed")
                completed += int(record.status == "completed")
                failed += int(record.status == "failed")
        if errors:
            raise RuntimeError("trace producer failed") from errors[0]
        return completed, failed
    finally:
        stop.set()
        queue.close(cancel=True)
        producer.join(timeout=5)
        try:
            pipeline.parts.executor.close()
        finally:
            pipeline.writer.close()
        if producer.is_alive():
            raise RuntimeError("trace producer did not stop")


def run_monitor_only(config: AppConfig) -> Path:
    if config.pipeline.mode != "monitor_only" or config.pipeline.partition != "validation":
        raise ValueError("monitor_only requires validation workload")
    if (
        not config.monitor.enabled
        or config.tier_selection.policy != "fixed"
        or config.admission.execution_mode != "sequential"
    ):
        raise ValueError("monitor_only requires monitor, static selector and sequential execution")
    return _run_stream(config)


def run_adaptive_sequential(config: AppConfig) -> Path:
    if (
        config.pipeline.mode != "adaptive_sequential"
        or config.pipeline.partition != "validation"
        or not config.monitor.enabled
        or config.admission.execution_mode != "sequential"
    ):
        raise ValueError(
            "adaptive_sequential requires monitor, validation and sequential execution"
        )
    return _run_stream(config)


def run_concurrent(config: AppConfig) -> Path:
    if (
        config.pipeline.mode != "concurrent"
        or config.pipeline.partition != "validation"
        or not config.monitor.enabled
    ):
        raise ValueError("concurrent requires monitor and validation workload")
    return _run_stream(config)


def _run_stream(config: AppConfig) -> Path:
    if config.workload.trace is None:
        raise ValueError("persist and select an arrival trace before execution")
    config, _ = resolve_training_data(config)
    assert config.split.manifest and config.models.checkpoint_index and config.workload.trace
    torch.set_num_threads(config.training.cpu_threads)
    records = load_trace(config.workload.trace, hash_file(config.split.manifest))
    with RunManager(config) as run:
        parts = assemble(config, run.logger)
        if config.uncertainty.artifact is not None:
            run.register_threshold(config.uncertainty.artifact)
        index = read_index(config.models.checkpoint_index)
        for tier in config.models.tier_portfolio:
            entry = index["checkpoints"][tier.value]
            run.register_checkpoint(
                Path(entry["directory"]) / "checkpoint_manifest.json", entry["checkpoint_id"]
            )
        threshold = run.directory / "artifacts/common_threshold.txt"
        write_immutable(threshold, parts.thresholds.definition())  # type: ignore[attr-defined]
        run.register_threshold(threshold)

        def queue_event(name: str, item: CandidateInput) -> None:
            run.logger.bind(item.candidate.trace_id, item.candidate.candidate_id).emit(
                module="queue", event_type=f"scheduler.queue.{name}", reason_code=name.upper()
            )

        queue = BoundedCandidateQueue(
            config.queue.capacity, config.queue.overflow, event=queue_event
        )
        counters = RuntimeCounters(
            config.monitor.estimator_window_s, config.monitor.estimator_max_samples
        )
        parts.queue = queue
        if config.pipeline.mode != "concurrent":
            parts.executor = ObservedExecutor(parts.executor, counters)
            parts.uncertainty = ObservedUncertainty(parts.uncertainty, counters)
        selector = ObservedSelector(parts.selector)
        parts.selector = selector
        admission = ObservedAdmission(parts.admission, counters)
        parts.admission = admission
        monitor = ResourceMonitor(
            [HostCollector(), NvidiaCollector(config.monitor.gpu_refresh_s)],
            counters,
            queue.stats,
            interval_s=config.monitor.interval_s,
            history_limit=config.monitor.history_limit,
            path=run.directory / "resources.jsonl",
            warning_fraction=config.monitor.overhead_warning_fraction,
            network={
                key: getattr(config.workload, key)
                for key in (
                    "network_available",
                    "network_bandwidth_bytes_per_s",
                    "network_latency_ms",
                )
            },
        )
        stop = threading.Event()
        source = TraceSource(config, run.manifest.run_id, records, stop)
        writer = ParquetRecordWriter(
            run.directory / "candidates.parquet", config.pipeline.record_batch_size
        )

        def snapshot(queue_length: int) -> ResourceSnapshot:
            current = monitor.current(queue_length)
            if config.pipeline.mode in {"adaptive_sequential", "concurrent"}:
                # Hardware stays timestamped; queue pressure is live and logged in decisions.
                stats = queue.stats()
                return current.model_copy(
                    update={
                        "queue_length": queue_length,
                        "oldest_wait_ms": stats["oldest_wait_ms"],
                        **counters.snapshot(),
                    }
                )
            return current

        engine = None
        if config.pipeline.mode == "concurrent":

            def admission_snapshot() -> ResourceSnapshot:
                # This executor owns every runtime lease; physical memory still includes all jobs.
                return snapshot(len(queue)).model_copy(
                    update={
                        "active_jobs_by_tier": (),
                        "active_jobs_total": 0,
                    }
                )

            engine = create_executor(config, admission_snapshot, run.logger)
            monitor.on_sample = engine.notify_resources
        pipeline = Pipeline(config, parts, run.logger, writer, snapshot=snapshot)
        monitor.start()
        try:
            if config.pipeline.mode in {"adaptive_sequential", "concurrent"}:
                deadline = time.monotonic() + config.pipeline.admission_timeout_s
                while monitor.current().snapshot_id is None:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("monitor startup timeout")
                    stop.wait(0.005)
            completed, failed = consume_stream(pipeline, source, queue, counters, stop, engine)
        finally:
            monitor.stop()
            if not writer.closed:  # Startup failed before consume_stream took ownership.
                try:
                    parts.executor.close()
                finally:
                    writer.close()
        overhead = monitor.overhead()
        stats = queue.stats()
        if overhead["failures"]:
            raise RuntimeError(f"monitor failures: {overhead['failures']}")
        if completed + failed + stats["rejected"] + stats["dropped"] != len(records):
            raise RuntimeError("candidate accounting mismatch")
        summary = {
            "completed": completed,
            "failed": failed,
            "executor": (
                {
                    "mode": config.admission.execution_mode,
                    "capacity": engine.concurrency,
                    "peak_active": engine.peak_active,
                    "submitted": engine.submitted,
                    "completed": engine.completed,
                    "failed": engine.failed,
                    "cancelled": engine.cancelled,
                }
                if engine
                else None
            ),
            "queue": stats,
            "workload": source.summary(),
            "monitor": overhead,
            "scheduler": {
                "calls": selector.calls,
                "selection_total_ms": selector.total_ms,
                "admission_calls": admission.calls,
                "admission_total_ms": admission.total_ms,
            },
            "availability": snapshot(len(queue)).model_dump(mode="json"),
            "trace_sha256": hash_file(config.workload.trace),
            "test_evaluated": False,
            "artifacts": {
                name: hash_file(run.directory / name)
                for name in ("resources.jsonl", "candidates.parquet")
            },
        }
        write_immutable(
            run.directory / f"metrics/{config.pipeline.mode}_summary.json", canonical(summary)
        )
        run.logger.emit(
            module="monitor",
            event_type="monitor.completed",
            reason_code="OVERHEAD_WARNING" if overhead["warning"] else "SAMPLED",
            payload={
                "samples": overhead["samples"],
                "sampling_wall_fraction": overhead["sampling_wall_fraction"],
                "completed": completed,
                "failed": failed,
            },
        )
    return run.directory
