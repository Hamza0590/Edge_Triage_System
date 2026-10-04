"""Bounded asynchronous whole-tier jobs with event-driven admission and thread workers.

All control methods belong to one asyncio loop. Only notify_resources() may be
called by a monitor thread. Models are loaded by the caller before submission.
Running native forwards cannot be interrupted safely: cancellation affects queued
jobs only, and shutdown always waits for launched work and reservation release.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from edge_triage.contracts import (
    AdmissionDecision,
    Contract,
    Identifier,
    ModelTier,
    NonNegative,
    PredictionResult,
    ResourceSnapshot,
    UncertaintyResult,
)
from edge_triage.runtime.interfaces import (
    CandidateInput,
    ImageUncertaintyPolicy,
    ModelAdapter,
    TierAdmissionPolicy,
    UncertaintyPolicy,
)
from edge_triage.uncertainty.sampling import SamplingError


@dataclass(frozen=True)
class InferenceJob:
    """One tier attempt, including optional UQ covered by the same admission lease."""

    job_id: str
    item: CandidateInput
    tier: ModelTier
    model: ModelAdapter
    uncertainty: UncertaintyPolicy | None = None
    on_launch: Callable[[AdmissionDecision | None, int], None] | None = None


class ExecutionResult(Contract):
    job_id: Identifier
    run_id: Identifier
    trace_id: Identifier
    candidate_id: Identifier
    tier: ModelTier
    status: Literal["completed", "failed", "cancelled"]
    prediction: PredictionResult | None = None
    uncertainty: UncertaintyResult | None = None
    queued_ns: int
    started_ns: int | None = None
    finished_ns: int
    queue_ms: NonNegative
    inference_ms: NonNegative = 0
    uq_ms: NonNegative = 0
    admission_ms: NonNegative = 0
    executor_ms: NonNegative = 0
    additional_forward_passes: int = 0
    failed_uq_completed: int = 0
    admission: AdmissionDecision | None = None
    error_type: str | None = None


class JobHandle:
    """Await without propagating caller cancellation into a running native forward."""

    def __init__(self, job_id: str, future: asyncio.Future[ExecutionResult]) -> None:
        self.job_id, self._future = job_id, future

    async def result(self) -> ExecutionResult:
        return await asyncio.shield(self._future)

    @property
    def done(self) -> bool:
        return self._future.done()


@dataclass
class _Pending:
    job: InferenceJob
    future: asyncio.Future[ExecutionResult]
    queued_ns: int
    deadline: float
    admission_ms: float = 0
    admission: AdmissionDecision | None = None
    reservation: str | None = None


@dataclass
class _Attempt:
    started_ns: int
    finished_ns: int
    prediction: PredictionResult | None = None
    uncertainty: UncertaintyResult | None = None
    inference_ms: float = 0
    uq_ms: float = 0
    additional_forward_passes: int = 0
    failed_uq_completed: int = 0
    error_type: str | None = None


def _infer(job: InferenceJob) -> _Attempt:
    result = _Attempt(time.monotonic_ns(), 0)
    try:
        start = time.perf_counter_ns()
        try:
            result.prediction = job.model.predict(job.item, job.tier)
        finally:
            result.inference_ms = (time.perf_counter_ns() - start) / 1e6
        if (
            result.prediction.candidate_id != job.item.candidate.candidate_id
            or result.prediction.tier != job.tier
        ):
            raise ValueError("executor prediction identity mismatch")
        if job.uncertainty is not None:
            start = time.perf_counter_ns()
            try:
                result.uncertainty = (
                    job.uncertainty.evaluate_image(result.prediction, job.item.image)
                    if isinstance(job.uncertainty, ImageUncertaintyPolicy)
                    else job.uncertainty.evaluate(result.prediction)
                )
            except SamplingError as error:
                result.additional_forward_passes = error.attempted
                result.failed_uq_completed = error.completed
                raise
            finally:
                result.uq_ms = (time.perf_counter_ns() - start) / 1e6
            result.additional_forward_passes = result.uncertainty.additional_forward_passes
            if (
                result.uncertainty.candidate_id != job.item.candidate.candidate_id
                or result.uncertainty.tier != job.tier
            ):
                raise ValueError("executor uncertainty identity mismatch")
    except BaseException as error:
        # A worker's BaseException must not tear down the event loop or leak a lease.
        result.error_type = type(error).__name__
    finally:
        result.finished_ns = time.monotonic_ns()
    return result


class FixedConcurrencyExecutor:
    """FIFO eligible-job dispatch, bounded pending work, and no tensor serialization.

    An optional lease policy can further restrict launches. The snapshot must count
    only jobs external to that policy's ledger, since Module07 adds owned leases.
    Pending jobs are revisited on submissions, completions, resource notifications,
    cancellation or the nearest admission deadline, never with a polling loop.
    """

    def __init__(
        self,
        concurrency: int,
        *,
        pending_capacity: int = 64,
        admission_timeout_s: float = 30,
        admission: TierAdmissionPolicy | None = None,
        snapshot: Callable[[], ResourceSnapshot] | None = None,
    ) -> None:
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
            or isinstance(pending_capacity, bool)
            or not isinstance(pending_capacity, int)
            or pending_capacity < 1
            or not math.isfinite(admission_timeout_s)
            or admission_timeout_s <= 0
        ):
            raise ValueError("positive capacities and finite positive timeout required")
        if (admission is None) != (snapshot is None):
            raise ValueError("admission and snapshot must be supplied together")
        self.concurrency, self.pending_capacity = concurrency, pending_capacity
        self.admission, self.snapshot = admission, snapshot
        self.admission_timeout_s = admission_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._wake = asyncio.Event()
        self._changed = asyncio.Condition()
        self._pending: deque[_Pending] = deque()
        self._active: dict[str, asyncio.Task[None]] = {}
        self._seen: set[str] = set()
        self._dispatcher: asyncio.Task[None] | None = None
        self._shutdown: asyncio.Task[None] | None = None
        self._closing = False
        self._broken = False
        self.peak_active = 0
        self.submitted = self.completed = self.failed = self.cancelled = 0

    def _bind(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("executor must be used on its owning event loop")
        if self._loop is None:
            self._loop = loop
        return loop

    async def submit(self, job: InferenceJob) -> JobHandle:
        loop = self._bind()
        # Validate IDs before accepting ownership so terminal records always serialize.
        ExecutionResult(
            job_id=job.job_id,
            run_id=job.item.candidate.run_id,
            trace_id=job.item.candidate.trace_id,
            candidate_id=job.item.candidate.candidate_id,
            tier=job.tier,
            status="cancelled",
            queued_ns=0,
            finished_ns=0,
            queue_ms=0,
        )
        async with self._changed:
            await self._changed.wait_for(
                lambda: self._closing or len(self._pending) < self.pending_capacity
            )
            if self._closing:
                raise RuntimeError("executor is shutting down")
            if job.job_id in self._seen:
                raise ValueError("duplicate job ID; use a distinct ID for each tier attempt")
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.concurrency, thread_name_prefix="edge-inference"
                )
                self._dispatcher = loop.create_task(self._dispatch())
            future: asyncio.Future[ExecutionResult] = loop.create_future()
            self._pending.append(
                _Pending(job, future, time.monotonic_ns(), loop.time() + self.admission_timeout_s)
            )
            self._seen.add(job.job_id)
            self.submitted += 1
            self._wake.set()
            return JobHandle(job.job_id, future)

    def notify_resources(self) -> None:
        """Thread-safe monitor hook; no admission occurs in the monitor thread."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._wake.set)

    async def cancel(self, job_id: str) -> bool:
        self._bind()
        async with self._changed:
            for pending in self._pending:
                if pending.job.job_id == job_id:
                    self._pending.remove(pending)
                    self._finish(pending, "cancelled")
                    self._changed.notify_all()
                    self._wake.set()
                    return True
            return False  # Running/completed/unknown jobs are never interrupted.

    async def drain(self) -> None:
        self._bind()
        async with self._changed:
            await self._changed.wait_for(lambda: not self._pending and not self._active)

    async def shutdown(self, *, cancel_pending: bool = False) -> None:
        loop = self._bind()
        if self._shutdown is None:
            self._closing = True
            self._shutdown = loop.create_task(self._close(cancel_pending))
        # Cancelling a caller cannot abandon launched native work or its lease.
        await asyncio.shield(self._shutdown)

    async def _close(self, cancel_pending: bool) -> None:
        async with self._changed:
            if cancel_pending:
                while self._pending:
                    self._finish(self._pending.popleft(), "cancelled")
            self._changed.notify_all()
            self._wake.set()
        await self.drain()
        self._wake.set()
        if self._dispatcher is not None:
            await self._dispatcher
        if self._pool is not None:
            self._pool.shutdown(wait=True)  # All submitted forwards already completed.
        if self._broken:
            raise RuntimeError("admission reservation release failed; executor stopped")

    def _finish(
        self,
        pending: _Pending,
        status: Literal["completed", "failed", "cancelled"],
        attempt: _Attempt | None = None,
        error_type: str | None = None,
    ) -> None:
        if pending.reservation is not None:
            token, pending.reservation = pending.reservation, None
            assert self.admission is not None
            try:
                self.admission.release(token, outcome=status)
            except BaseException as error:
                status, error_type = "failed", type(error).__name__
                # Stop new launches when ledger cleanup cannot be established.
                self._closing = True
                self._broken = True
        now = time.monotonic_ns()
        candidate = pending.job.item.candidate
        result = ExecutionResult(
            job_id=pending.job.job_id,
            run_id=candidate.run_id,
            trace_id=candidate.trace_id,
            candidate_id=candidate.candidate_id,
            tier=pending.job.tier,
            status=status,
            prediction=attempt.prediction if attempt else None,
            uncertainty=attempt.uncertainty if attempt else None,
            queued_ns=pending.queued_ns,
            started_ns=attempt.started_ns if attempt else None,
            finished_ns=now,
            queue_ms=((attempt.started_ns if attempt else now) - pending.queued_ns) / 1e6,
            inference_ms=attempt.inference_ms if attempt else 0,
            uq_ms=attempt.uq_ms if attempt else 0,
            admission_ms=pending.admission_ms,
            executor_ms=(max(0, now - attempt.finished_ns) / 1e6 if attempt else 0),
            additional_forward_passes=attempt.additional_forward_passes if attempt else 0,
            failed_uq_completed=attempt.failed_uq_completed if attempt else 0,
            admission=pending.admission,
            error_type=error_type or (attempt.error_type if attempt else None),
        )
        pending.future.set_result(result)
        self.completed += int(status == "completed")
        self.failed += int(status == "failed")
        self.cancelled += int(status == "cancelled")

    async def _run(self, pending: _Pending) -> None:
        assert self._loop is not None and self._pool is not None
        try:
            attempt = await self._loop.run_in_executor(self._pool, _infer, pending.job)
            self._finish(pending, "failed" if attempt.error_type else "completed", attempt)
        except Exception as error:
            self._finish(pending, "failed", error_type=type(error).__name__)
        finally:
            self._active.pop(pending.job.job_id)
            async with self._changed:
                self._changed.notify_all()
                self._wake.set()

    async def _dispatch(self) -> None:
        assert self._loop is not None
        while True:
            self._wake.clear()
            # One pass in arrival order: an ineligible tier cannot block eligible tiers.
            for _ in range(len(self._pending)):
                pending = self._pending.popleft()
                if self._broken:
                    self._finish(pending, "failed", error_type="ReservationReleaseError")
                    continue
                if self._loop.time() >= pending.deadline:
                    self._finish(pending, "failed", error_type="AdmissionTimeoutError")
                    continue
                if len(self._active) >= self.concurrency:
                    self._pending.append(pending)
                    continue
                if self.admission is not None:
                    assert self.snapshot is not None
                    start = time.perf_counter_ns()
                    try:
                        decision = self.admission.admit(
                            pending.job.item.candidate, pending.job.tier, self.snapshot()
                        )
                        pending.admission = decision
                        pending.reservation = decision.reservation_id
                        if decision.candidate_id != pending.job.item.candidate.candidate_id:
                            raise ValueError("admission changed candidate identity")
                        if decision.outcome != "admitted" and pending.reservation is not None:
                            raise ValueError("non-admitted job owns a reservation")
                        if decision.outcome == "admitted" and pending.reservation is None:
                            raise ValueError("lease-aware admission did not reserve a job")
                    except Exception as error:
                        pending.admission_ms += (time.perf_counter_ns() - start) / 1e6
                        self._finish(pending, "failed", error_type=type(error).__name__)
                        continue
                    pending.admission_ms += (time.perf_counter_ns() - start) / 1e6
                    if decision.outcome == "delayed":
                        self._pending.append(pending)
                        continue
                    if decision.outcome == "rejected":
                        self._finish(pending, "failed", error_type="AdmissionRejectedError")
                        continue
                try:
                    if pending.job.on_launch is not None:
                        pending.job.on_launch(pending.admission, len(self._active) + 1)
                except Exception as error:
                    self._finish(pending, "failed", error_type=type(error).__name__)
                    continue
                self._active[pending.job.job_id] = self._loop.create_task(self._run(pending))
                self.peak_active = max(self.peak_active, len(self._active))
            async with self._changed:
                self._changed.notify_all()
            if self._closing and not self._pending and not self._active:
                return
            timeout = (
                max(0, min(p.deadline for p in self._pending) - self._loop.time())
                if self._pending
                else None
            )
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                pass


class SequentialExecutor(FixedConcurrencyExecutor):
    def __init__(
        self,
        *,
        pending_capacity: int = 64,
        admission_timeout_s: float = 30,
        admission: TierAdmissionPolicy | None = None,
        snapshot: Callable[[], ResourceSnapshot] | None = None,
    ) -> None:
        super().__init__(
            1,
            pending_capacity=pending_capacity,
            admission_timeout_s=admission_timeout_s,
            admission=admission,
            snapshot=snapshot,
        )


class DynamicConcurrencyExecutor(FixedConcurrencyExecutor):
    def __init__(
        self,
        max_concurrency: int,
        admission: TierAdmissionPolicy,
        snapshot: Callable[[], ResourceSnapshot],
        *,
        pending_capacity: int = 64,
        admission_timeout_s: float = 30,
    ) -> None:
        super().__init__(
            max_concurrency,
            pending_capacity=pending_capacity,
            admission_timeout_s=admission_timeout_s,
            admission=admission,
            snapshot=snapshot,
        )
