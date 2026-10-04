"""Configuration-driven async execution with one owner for analytical output."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from edge_triage.config import AppConfig
from edge_triage.contracts import AdmissionDecision, Candidate, ModelTier, ResourceSnapshot
from edge_triage.runtime.executors import (
    DynamicConcurrencyExecutor,
    FixedConcurrencyExecutor,
    SequentialExecutor,
)
from edge_triage.scheduling.policies import (
    FixedConcurrencyAdmissionPolicy,
    MemoryContentionAwareAdmissionPolicy,
    PolicyContext,
)
from edge_triage.scheduling.profiles import scheduler_profiles


class FixedMemoryAdmission(FixedConcurrencyAdmissionPolicy):
    """Constant-cap baseline with the same physical-memory guard as dynamic admission."""

    def admit(
        self, candidate: Candidate, selected_tier: ModelTier, resources: ResourceSnapshot
    ) -> AdmissionDecision:
        start = time.perf_counter_ns()
        with self.lock:
            ctx = self.context
            data, before = ctx.inputs(resources, ctx.clock()), self.state()
            profile = ctx.profiles.get(selected_tier)
            free = data["free_bytes"]
            reason = None
            if selected_tier not in ctx.portfolio or profile is None:
                reason = "NO_FEASIBLE_TIER"
            elif free is None:
                reason = "INSUFFICIENT_HISTORY"
            elif (
                free - before["reserved_bytes"] - profile.memory_bytes
                < ctx.constraints.memory_margin_bytes
            ):
                reason = "MEMORY_HEADROOM_LOW"
            else:
                reason = self._reason(selected_tier, data)
            return self._finish(
                candidate,
                selected_tier,
                resources,
                start,
                data,
                before,
                reason,
                profile.memory_bytes if profile else 0,
                "fixed_memory",
            )


def create_executor(
    config: AppConfig, snapshot: Callable[[], ResourceSnapshot], logger: Any = None
) -> FixedConcurrencyExecutor:
    """All three modes share profiles and memory/per-tier constraints."""
    context = PolicyContext(
        config.models.tier_portfolio,
        scheduler_profiles(config),
        config.scheduler.constraints,
        device=config.scheduler.device,
        logger=logger,
    )
    mode = config.admission.execution_mode
    capacity = (
        1
        if mode == "sequential"
        else config.admission.fixed_concurrency
        if mode == "fixed_concurrency"
        else config.admission.max_concurrency
    )
    if capacity > context.constraints.max_total_active:
        raise ValueError("executor capacity exceeds scheduler total cap")
    if mode == "dynamic_concurrency" and config.scheduler.contention_profile is not None:
        from edge_triage.scheduling.contention import load_contention_cap

        capacity = min(capacity, load_contention_cap(config))
        context.constraints = context.constraints.model_copy(update={"max_total_active": capacity})
    kwargs: dict[str, Any] = dict(
        pending_capacity=config.queue.capacity,
        admission_timeout_s=config.pipeline.admission_timeout_s,
    )
    if mode == "dynamic_concurrency":
        return DynamicConcurrencyExecutor(
            capacity, MemoryContentionAwareAdmissionPolicy(context), snapshot, **kwargs
        )
    policy = FixedMemoryAdmission(context, capacity)
    if mode == "sequential":
        return SequentialExecutor(admission=policy, snapshot=snapshot, **kwargs)
    return FixedConcurrencyExecutor(capacity, admission=policy, snapshot=snapshot, **kwargs)


async def consume_jobs(
    get_entry: Callable[[], Any],
    process: Callable[..., Any],
    finish: Callable[..., None],
    executor: FixedConcurrencyExecutor,
    window: int,
) -> tuple[int, int]:
    """Bounded candidates in flight; get_entry blocks on the producer's condition queue."""
    active: set[asyncio.Task[Any]] = set()
    completed = failed = 0

    async def one(item: Any, entry: Any) -> None:
        nonlocal completed, failed
        record = await process(item, entry, executor)
        finish(item, record)
        completed += int(record.status == "completed")
        failed += int(record.status == "failed")

    try:
        while True:
            if len(active) >= window:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            pair = await asyncio.to_thread(get_entry)
            if pair is None:
                break
            active.add(asyncio.create_task(one(*pair)))
        if active:
            await asyncio.gather(*active)
        return completed, failed
    finally:
        # Never abandon workers/leases if source/output fails.
        await executor.shutdown(cancel_pending=True)
        if active:
            await asyncio.gather(*active, return_exceptions=True)
