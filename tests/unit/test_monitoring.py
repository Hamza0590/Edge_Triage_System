import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from edge_triage.config import QueueConfig, WorkloadConfig
from edge_triage.contracts import Candidate, ModelTier, ResourceSnapshot
from edge_triage.data.artifacts import canonical
from edge_triage.experiments.workloads import PROFILES, generate
from edge_triage.monitoring.collectors import HostCollector, NvidiaCollector
from edge_triage.monitoring.estimators import RollingWindow, RuntimeCounters
from edge_triage.monitoring.monitor import ResourceMonitor
from edge_triage.runtime.bounded_queue import BoundedCandidateQueue, QueueClosed, QueueOverflow
from edge_triage.runtime.interfaces import CandidateInput


def item(index):
    return CandidateInput(
        Candidate(
            run_id="r",
            trace_id=f"t{index}",
            candidate_id=f"c{index}",
            dataset_index=index,
            arrival_monotonic_ns=0,
        ),
        None,
        0,
        10,
    )


def test_rolling_known_timestamps_and_empty_history():
    window = RollingWindow(5, 3)
    assert window.summary(0)["mean"] is None
    window.add(10, 0)
    assert window.summary(0)["rate"] is None
    window.add(20, 2)
    assert window.summary(2) == {"samples": 2, "mean": 15.0, "rate": 0.5, "truncated": False}
    window.add(30, 3)
    window.add(40, 4)
    assert window.summary(4)["truncated"]
    assert window.summary(4)["mean"] is None
    assert window.summary(10)["samples"] == 0
    with pytest.raises(ValueError):
        window.add(0, 1)


def test_thread_safe_job_counters():
    counters = RuntimeCounters(5, 10000)

    def work(_):
        for _ in range(100):
            counters.started(ModelTier.TINY)
            counters.finished(ModelTier.TINY, 2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))
    result = counters.snapshot()
    assert result["active_jobs_total"] == 0
    assert result["latency_samples"] == 800
    assert result["recent_latency_ms"] == 2
    assert result["arrival_rate_per_s"] is None
    with pytest.raises(ValueError, match="underflow"):
        counters.finished(ModelTier.TINY, 2)


def test_admission_uses_live_counts_not_stale_sampler():
    from edge_triage.contracts import TierJobs, utc_now
    from edge_triage.runtime.reference import SequentialAdmissionPolicy
    from edge_triage.runtime.streaming import ObservedAdmission

    counters = RuntimeCounters()
    admission = ObservedAdmission(SequentialAdmissionPolicy(), counters)
    stale = ResourceSnapshot(
        timestamp_utc=utc_now(),
        monotonic_ns=0,
        cpu_percent=None,
        available_ram_bytes=None,
        queue_length=1,
        arrival_rate_per_s=None,
        active_jobs_by_tier=(TierJobs(tier=ModelTier.TINY, count=1),),
    )
    assert admission.decide(item(0).candidate, stale).outcome == "admitted"
    counters.started(ModelTier.TINY)
    assert admission.decide(item(0).candidate, stale).outcome == "delayed"
    counters.finished(ModelTier.TINY, 1)


class MissingCollector:
    def collect(self):
        raise OSError("no device")


def test_monitor_missing_lifecycle_bounded_history_and_persistence(tmp_path):
    path = tmp_path / "samples.jsonl"
    monitor = ResourceMonitor(
        [MissingCollector()],
        RuntimeCounters(),
        lambda: {"length": 2, "oldest_wait_ms": 1},
        interval_s=0.005,
        history_limit=3,
        path=path,
    )
    assert monitor.current().availability["monitor"] == "warming_up"
    monitor.stop()
    monitor.start()
    first_thread = monitor.thread
    monitor.start()
    assert first_thread is monitor.thread
    time.sleep(0.05)
    monitor.stop()
    monitor.stop()
    assert len(monitor.history) == 3
    assert monitor.current().cpu_percent is None
    assert monitor.current().gpu_utilization_percent is None
    assert monitor.current().arrival_rate_per_s is None
    assert monitor.current().availability["MissingCollector"] == "unavailable:OSError"
    assert monitor.current().queue_length == 2
    samples = [ResourceSnapshot.model_validate_json(line) for line in path.read_text().splitlines()]
    assert len(samples) == monitor.sample_count >= 3
    assert len({s.snapshot_id for s in samples}) == len(samples)
    monitor.start()
    time.sleep(0.02)
    monitor.stop()
    assert monitor.sample_count > len(samples)


def test_collector_fallback_and_cache(monkeypatch):
    calls = []

    def missing(*args, **kwargs):
        calls.append(1)
        raise FileNotFoundError()

    monkeypatch.setattr("edge_triage.monitoring.collectors.subprocess.run", missing)
    collector = NvidiaCollector(100)
    assert "unavailable" in collector.collect()["availability"]["gpu"]
    collector.collect()
    assert len(calls) == 1
    host = HostCollector()
    assert host.collect()["cpu_percent"] is None
    assert host.collect()["process_rss_bytes"] > 0


def test_monitor_overhead_warning():
    class Slow:
        def collect(self):
            time.sleep(0.01)
            return {}

    monitor = ResourceMonitor(
        [Slow()], RuntimeCounters(), lambda: {"length": 0, "oldest_wait_ms": 0}, interval_s=0.005
    )
    monitor.start()
    time.sleep(0.06)
    monitor.stop()
    assert monitor.overhead()["warning"]


def test_queue_fifo_blocking_lifecycle_and_shutdown():
    events = []
    queue = BoundedCandidateQueue(1, event=lambda name, item: events.append(name))
    queue.put(item(0))
    put_done = threading.Event()
    producer = threading.Thread(target=lambda: (queue.put(item(1)), put_done.set()))
    producer.start()
    assert not put_done.wait(0.02)
    first = queue.get()
    assert first.candidate.candidate_id == "c0"
    queue.done(first)
    assert put_done.wait(1)
    queue.close()
    second = queue.get()
    queue.done(second, failed=True)
    producer.join()
    with pytest.raises(QueueClosed):
        queue.get()
    stats = queue.stats()
    assert stats["peak_length"] == 1 and stats["completed"] == stats["failed"] == 1
    assert stats["blocked_puts"] == 1 and stats["active"] == 0
    assert "backpressure" in events


@pytest.mark.parametrize("policy,outcome", [("reject", "rejected"), ("drop", "dropped")])
def test_controlled_overflow(policy, outcome):
    with pytest.raises(ValidationError):
        QueueConfig(overflow=policy)
    queue = BoundedCandidateQueue(1, policy)
    queue.put(item(0))
    with pytest.raises(QueueOverflow, match=outcome):
        queue.put(item(1))
    assert queue.stats()[outcome] == 1
    queue.close(cancel=True)
    assert queue.stats()["cancelled"] == 1


def test_shutdown_unblocks_producer_and_consumer():
    queue = BoundedCandidateQueue(1)
    queue.put(item(0))
    error = []

    def blocked():
        try:
            queue.put(item(1))
        except QueueClosed:
            error.append("closed")

    thread = threading.Thread(target=blocked)
    thread.start()
    time.sleep(0.01)
    queue.close(cancel=True)
    thread.join(1)
    assert not thread.is_alive() and error == ["closed"]
    with pytest.raises(QueueClosed):
        queue.get()


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("process", ["periodic", "poisson"])
def test_trace_determinism_repeated_rows_and_no_test(profile, process):
    rows = [{"candidate_id": f"c{i}", "index_no": i, "split": "validation"} for i in range(3)]
    spec = WorkloadConfig(count=10, profile=profile, arrival_process=process)
    records = generate(rows, spec)

    def encoded(rs):
        return canonical([r.model_dump(mode="json") for r in rs])

    assert encoded(records) == encoded(generate(rows, spec))
    assert len({r.candidate_id for r in records}) == 10
    assert records[3].repetition == 1 and records[9].repetition == 3
    assert all(not r.independent_classification_sample for r in records)
    assert [r.relative_arrival_seconds for r in records] == sorted(
        r.relative_arrival_seconds for r in records
    )
    if profile == "backlog":
        assert records[-1].relative_arrival_seconds == 0
    with pytest.raises(ValueError, match="validation"):
        generate([{"candidate_id": "test", "split": "test"}], spec)
