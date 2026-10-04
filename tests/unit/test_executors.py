"""Synthetic executor tests: no frozen inputs, labels, training or timing claims."""

import asyncio
import threading
import time

import pytest
import torch

from edge_triage.contracts import (
    Candidate,
    ModelTier,
    PredictionResult,
    ResourceSnapshot,
    TierLatency,
    utc_now,
)
from edge_triage.runtime.executors import (
    DynamicConcurrencyExecutor,
    FixedConcurrencyExecutor,
    InferenceJob,
    SequentialExecutor,
)
from edge_triage.runtime.interfaces import CandidateInput
from edge_triage.runtime.reference import NoUncertaintyPolicy
from edge_triage.scheduling.policies import MemoryContentionAwareAdmissionPolicy, PolicyContext
from edge_triage.scheduling.profiles import SchedulerProfile
from edge_triage.scheduling.settings import SchedulerConstraints
from edge_triage.uncertainty.sampling import SamplingError

T, M, L = ModelTier


def snapshot(**updates):
    values = dict(
        timestamp_utc=utc_now(),
        monotonic_ns=time.monotonic_ns(),
        cpu_percent=10,
        available_ram_bytes=10000,
        queue_length=0,
        arrival_rate_per_s=None,
        active_jobs_total=0,  # This executor's owned jobs already exist in the lease ledger.
        recent_latency_by_tier=tuple(TierLatency(tier=t, mean_ms=1, samples=10) for t in ModelTier),
    )
    values.update(updates)
    return ResourceSnapshot(**values)


class Ledger(MemoryContentionAwareAdmissionPolicy):
    def __init__(self, cap=4, tier_cap=None):
        constraints = SchedulerConstraints(
            max_total_active=cap,
            per_tier_caps={t: tier_cap or cap for t in ModelTier},
            memory_margin_bytes=50,
            hysteresis_s=0,
            cooldown_s=0,
        )
        profiles = {
            t: SchedulerProfile(
                tier=t,
                profile_id=t.value,
                checkpoint_id=t.value,
                latency_ms=1,
                latency_p95_ms=1,
                memory_bytes=100,
                memory_basis="synthetic_test",
            )
            for t in ModelTier
        }
        super().__init__(PolicyContext(tuple(ModelTier), profiles, constraints))
        self.releases = []
        self.calls = 0
        self.peak_bytes = 0

    def admit(self, candidate, selected_tier, resources):
        self.calls += 1
        result = super().admit(candidate, selected_tier, resources)
        self.peak_bytes = max(self.peak_bytes, self.state()["reserved_bytes"])
        return result

    def release(self, token, *, outcome="completed"):
        assert token not in [t for t, _ in self.releases], "double release"
        self.releases.append((token, outcome))
        super().release(token, outcome=outcome)


class Model:
    def __init__(self, *, hold=None, entered=None, failures=(), slow=()):
        self.hold, self.entered = hold, entered
        self.failures, self.slow = failures, slow
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.ids = []

    def predict(self, item, tier):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.entered:
                self.entered.set()
            if self.hold:
                assert self.hold.wait(5), "test release missing"
            if item.candidate.dataset_index in self.slow:
                time.sleep(0.04)
            if item.candidate.dataset_index in self.failures:
                raise RuntimeError("injected forward failure")
            with self.lock:
                self.ids.append(item.candidate.candidate_id)
            return PredictionResult(
                candidate_id=item.candidate.candidate_id,
                tier=tier,
                checkpoint_id="test",
                logit=0,
                p_real=0.5,
                inference_ms=0,
            )
        finally:
            with self.lock:
                self.active -= 1


def job(index, model, tier=T, uncertainty=None):
    item = CandidateInput(
        Candidate(
            run_id="run",
            trace_id=f"trace{index}",
            candidate_id=f"candidate{index}",
            dataset_index=index,
            arrival_monotonic_ns=time.monotonic_ns(),
        ),
        torch.zeros(3, 30, 30),
        index % 2,
        120000,
    )
    return InferenceJob(f"job{index}", item, tier, model, uncertainty)


def backend(mode, cap=4, **kwargs):
    if mode == "sequential":
        return SequentialExecutor(**kwargs)
    if mode == "fixed":
        return FixedConcurrencyExecutor(cap, **kwargs)
    return DynamicConcurrencyExecutor(cap, Ledger(cap), snapshot, **kwargs)


async def entered(event):
    assert await asyncio.to_thread(event.wait, 5)


@pytest.mark.parametrize("mode", ["sequential", "fixed", "dynamic"])
def test_shared_contract_long_trace_failures_and_identity(mode):
    async def scenario():
        executor = backend(mode, pending_capacity=7)
        model = Model(failures={13, 67, 129})
        try:
            handles = [
                await executor.submit(job(i, model, tuple(ModelTier)[i % 3])) for i in range(300)
            ]
            await executor.drain()
            results = await asyncio.gather(*(h.result() for h in reversed(handles)))
            assert len({r.job_id for r in results}) == 300
            for result in results:
                i = int(result.job_id[3:])
                assert (result.candidate_id, result.trace_id) == (f"candidate{i}", f"trace{i}")
                assert result.status == ("failed" if i in model.failures else "completed")
                assert result.tier == tuple(ModelTier)[i % 3]
                assert result.queued_ns <= result.started_ns <= result.finished_ns
            assert executor.submitted == executor.completed + executor.failed == 300
            assert model.active == 0 and model.peak <= executor.concurrency
            if mode == "dynamic":
                assert not executor.admission.reservations
                assert len(executor.admission.releases) == 300
        finally:
            await executor.shutdown()
        await executor.shutdown()
        with pytest.raises(RuntimeError, match="shutting down"):
            await executor.submit(job(301, model))

    asyncio.run(scenario())


@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_fixed_levels_reach_cap_and_complete_out_of_order(cap):
    async def scenario():
        release = threading.Event()
        start = threading.Event()
        model = Model(hold=release, entered=start, slow={0})
        executor = FixedConcurrencyExecutor(cap)
        try:
            handles = [await executor.submit(job(i, model)) for i in range(cap + 2)]
            await entered(start)
            # Synchronize with all worker entries, not a performance assertion.
            for _ in range(500):
                if model.active == cap:
                    break
                await asyncio.sleep(0.002)
            assert model.active == cap
            release.set()
            results = await asyncio.gather(*(h.result() for h in handles))
            assert all(r.status == "completed" for r in results)
            assert executor.peak_active == model.peak == cap
            if cap > 1:
                assert model.ids.index("candidate1") < model.ids.index("candidate0")
            assert [r.candidate_id for r in results] == [f"candidate{i}" for i in range(cap + 2)]
        finally:
            release.set()
            await executor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["sequential", "fixed", "dynamic"])
def test_cancel_backpressure_waiter_cancellation_and_shutdown(mode):
    async def scenario():
        release, start = threading.Event(), threading.Event()
        model = Model(hold=release, entered=start)
        executor = backend(mode, cap=1, pending_capacity=1)
        try:
            first = await executor.submit(job(0, model))
            await entered(start)
            second = await executor.submit(job(1, model))
            blocked = asyncio.create_task(executor.submit(job(2, model)))
            await asyncio.sleep(0.01)
            assert not blocked.done()
            assert not await executor.cancel("job0")
            assert await executor.cancel("job1")
            assert (await second.result()).status == "cancelled"
            third = await blocked
            waiter = asyncio.create_task(first.result())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not first.done
            closing = asyncio.create_task(executor.shutdown(cancel_pending=True))
            await asyncio.sleep(0.01)
            assert not closing.done() and not first.done
            assert (await third.result()).status == "cancelled"
            release.set()
            await closing
            assert (await first.result()).status == "completed"
            assert executor.submitted == 3 and executor.cancelled == 2
        finally:
            release.set()
            await executor.shutdown()

    asyncio.run(scenario())


def test_pressure_notification_no_polling_memory_cap_and_recovery():
    async def scenario():
        pressure = {"available_ram_bytes": 149}
        ledger = Ledger(8)
        executor = DynamicConcurrencyExecutor(8, ledger, lambda: snapshot(**pressure))
        model = Model()
        try:
            handle = await executor.submit(job(0, model))
            await asyncio.sleep(0.02)
            calls = ledger.calls
            assert calls == 1 and not handle.done
            await asyncio.sleep(0.03)
            assert ledger.calls == calls  # No periodic admission retry.
            pressure["available_ram_bytes"] = 250  # Exactly two 100-byte leases + margin.
            await asyncio.to_thread(executor.notify_resources)
            assert (await asyncio.wait_for(handle.result(), 2)).status == "completed"
            handles = [await executor.submit(job(i, model)) for i in range(1, 30)]
            await asyncio.gather(*(h.result() for h in handles))
            assert ledger.peak_bytes <= 200 and not ledger.reservations
            assert len(ledger.releases) == 30
        finally:
            await executor.shutdown()

    asyncio.run(scenario())


def test_pressure_reduces_admissions_without_cancelling_running_jobs():
    async def scenario():
        release, start = threading.Event(), threading.Event()
        model = Model(hold=release, entered=start)
        pressure = {"cpu_percent": 10}
        ledger = Ledger(4)
        executor = DynamicConcurrencyExecutor(4, ledger, lambda: snapshot(**pressure))
        try:
            first = await executor.submit(job(0, model))
            await entered(start)
            pressure["cpu_percent"] = 99
            second = await executor.submit(job(1, model))
            executor.notify_resources()
            await asyncio.sleep(0.02)
            assert ledger.capacity == 1 and not first.done and not second.done
            release.set()
            assert (await first.result()).status == "completed"
            await asyncio.sleep(0.02)
            assert not second.done
            pressure["cpu_percent"] = 10
            executor.notify_resources()
            assert (await asyncio.wait_for(second.result(), 2)).status == "completed"
        finally:
            release.set()
            await executor.shutdown(cancel_pending=True)

    asyncio.run(scenario())


def test_timeout_duplicate_identity_and_failed_uq_exact_release():
    class BrokenUQ:
        def evaluate(self, prediction):
            raise SamplingError(3, 2, 0.1)

    async def scenario():
        ledger = Ledger()
        pressure = {"available_ram_bytes": 0}
        executor = DynamicConcurrencyExecutor(
            4, ledger, lambda: snapshot(**pressure), admission_timeout_s=0.03
        )
        model = Model()
        try:
            waiting = await executor.submit(job(0, model))
            with pytest.raises(ValueError, match="duplicate"):
                await executor.submit(job(0, model))
            result = await waiting.result()
            assert result.error_type == "AdmissionTimeoutError" and not ledger.releases
            pressure["available_ram_bytes"] = 10000
            failing = await executor.submit(job(1, model, uncertainty=BrokenUQ()))
            result = await failing.result()
            assert result.status == "failed" and result.prediction is not None
            assert result.error_type == "SamplingError"
            assert (result.additional_forward_passes, result.failed_uq_completed) == (3, 2)
            assert result.uq_ms > 0 and not ledger.reservations
            assert [outcome for _, outcome in ledger.releases] == ["failed"]
            good = await executor.submit(job(2, model, uncertainty=NoUncertaintyPolicy()))
            assert (await good.result()).status == "completed"
        finally:
            await executor.shutdown()

    asyncio.run(scenario())


def test_lease_survives_uq_and_eligible_tier_bypasses_delayed_tier():
    release, start = threading.Event(), threading.Event()

    class HeldUQ(NoUncertaintyPolicy):
        def evaluate(self, prediction):
            start.set()
            assert release.wait(5)
            return super().evaluate(prediction)

    async def scenario():
        ledger = Ledger(4, tier_cap=1)
        executor = DynamicConcurrencyExecutor(4, ledger, snapshot)
        model = Model()
        try:
            one = await executor.submit(job(0, model, uncertainty=HeldUQ()))
            await entered(start)
            assert len(ledger.reservations) == 1 and not ledger.releases
            two = await executor.submit(job(1, model))
            three = await executor.submit(job(2, model, M))
            assert (await asyncio.wait_for(three.result(), 2)).status == "completed"
            assert not one.done and not two.done
            release.set()
            assert (await one.result()).status == (await two.result()).status == "completed"
            assert len(ledger.releases) == 3 and not ledger.reservations
        finally:
            release.set()
            await executor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("args", [(0,), (True,), (1.5,)])
def test_invalid_capacity(args):
    with pytest.raises(ValueError):
        FixedConcurrencyExecutor(*args)


def test_shutdown_caller_cancellation_does_not_release_running_lease():
    async def scenario():
        release, start = threading.Event(), threading.Event()
        ledger = Ledger(1)
        executor = DynamicConcurrencyExecutor(1, ledger, snapshot)
        try:
            handle = await executor.submit(job(0, Model(hold=release, entered=start)))
            await entered(start)
            closing = asyncio.create_task(executor.shutdown())
            await asyncio.sleep(0)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert len(ledger.reservations) == 1 and not ledger.releases
            release.set()
            await executor.shutdown()
            assert (await handle.result()).status == "completed"
            assert len(ledger.releases) == 1 and not ledger.reservations
        finally:
            release.set()
            await executor.shutdown()

    asyncio.run(scenario())


def test_release_failure_stops_launches_and_reports_shutdown_error():
    class BrokenRelease(Ledger):
        def release(self, token, *, outcome="completed"):
            raise RuntimeError("ledger unavailable")

    async def scenario():
        executor = DynamicConcurrencyExecutor(1, BrokenRelease(1), snapshot)
        model = Model()
        handles = [await executor.submit(job(i, model)) for i in range(3)]
        results = await asyncio.gather(*(h.result() for h in handles))
        assert all(r.status == "failed" for r in results)
        assert model.ids == ["candidate0"]
        with pytest.raises(RuntimeError, match="reservation release failed"):
            await executor.shutdown()

    asyncio.run(scenario())


def test_bad_prediction_identity_is_isolated_and_releases_lease():
    class WrongIdentity(Model):
        def predict(self, item, tier):
            return super().predict(item, tier).model_copy(update={"candidate_id": "wrong"})

    async def scenario():
        ledger = Ledger(1)
        executor = DynamicConcurrencyExecutor(1, ledger, snapshot)
        try:
            bad = await executor.submit(job(0, WrongIdentity()))
            good = await executor.submit(job(1, Model()))
            assert (await bad.result()).status == "failed"
            assert (await good.result()).status == "completed"
            assert [outcome for _, outcome in ledger.releases] == ["failed", "completed"]
            assert not ledger.reservations
        finally:
            await executor.shutdown()

    asyncio.run(scenario())
