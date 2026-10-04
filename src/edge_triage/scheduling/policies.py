"""Bounded deterministic rules and atomic leases; no I/O or inference in decisions."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar

from edge_triage.contracts import (
    AdmissionDecision,
    Candidate,
    ModelTier,
    ResourceSnapshot,
    SchedulingDecision,
)
from edge_triage.event_logging import EventLogger
from edge_triage.runtime.interfaces import AdmissionPolicy, TierSelectionPolicy
from edge_triage.scheduling.profiles import SchedulerProfile
from edge_triage.scheduling.settings import SchedulerConstraints

ORDER = {tier: i for i, tier in enumerate(ModelTier)}
Decision = TypeVar("Decision", SchedulingDecision, AdmissionDecision)


class PolicyContext:
    def __init__(
        self,
        portfolio: tuple[ModelTier, ...],
        profiles: Mapping[ModelTier, SchedulerProfile],
        constraints: SchedulerConstraints,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        clock: Callable[[], int] = time.monotonic_ns,
        logger: EventLogger | None = None,
    ) -> None:
        if not portfolio or len(set(portfolio)) != len(portfolio):
            raise ValueError("portfolio must contain unique enabled tiers")
        self.portfolio = tuple(sorted(portfolio, key=ORDER.__getitem__))
        self.profiles = {t: profiles[t] for t in self.portfolio if t in profiles}
        if any(t != p.tier for t, p in self.profiles.items()):
            raise ValueError("profile tier mismatch")
        self.constraints, self.device, self.clock, self.logger = constraints, device, clock, logger

    def inputs(self, snapshot: ResourceSnapshot, now: int) -> dict[str, Any]:
        c = self.constraints
        fresh = 0 <= now - snapshot.monotonic_ns <= c.max_snapshot_age_s * 1e9
        observed = snapshot.gpu_observed_monotonic_ns
        gpu_fresh = fresh and observed is not None and 0 <= now - observed <= c.max_gpu_age_s * 1e9
        free = (
            snapshot.available_ram_bytes
            if self.device == "cpu"
            else (snapshot.gpu_memory_free_bytes if gpu_fresh else None)
        )
        history = {p.tier: p for p in snapshot.recent_latency_by_tier}
        ratios = {
            t.value: float(history[t].mean_ms or 0) / p.latency_ms
            for t, p in self.profiles.items()
            if fresh
            and t in history
            and history[t].samples >= c.min_latency_samples
            and history[t].mean_ms is not None
        }
        return {
            "now_ns": now,
            "snapshot_ns": snapshot.monotonic_ns,
            "fresh": fresh,
            "gpu_fresh": gpu_fresh,
            "free_bytes": free if fresh else None,
            "cpu_percent": snapshot.cpu_percent if fresh else None,
            "gpu_percent": snapshot.gpu_utilization_percent if gpu_fresh else None,
            "queue_length": snapshot.queue_length,
            "oldest_wait_ms": snapshot.oldest_wait_ms,
            "active_jobs": {r.tier.value: r.count for r in snapshot.active_jobs_by_tier},
            "active_jobs_total": snapshot.active_jobs_total,
            "latency_ratios": ratios,
            "latency_history": [r.model_dump(mode="json") for r in snapshot.recent_latency_by_tier],
        }

    def record(
        self,
        policy: str,
        candidate: Candidate,
        snapshot: ResourceSnapshot,
        decision: Decision,
        started: int,
        inputs: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
        evaluated: dict[str, Any],
    ) -> Decision:
        explanation = {
            "policy": policy,
            "version": "1",
            "snapshot_id": snapshot.snapshot_id,
            "portfolio": [t.value for t in self.portfolio],
            "candidate_id": candidate.candidate_id,
            "constraints": self.constraints.model_dump(mode="json"),
            "inputs": inputs,
            "evaluated": evaluated,
            "profiles": {t.value: p.model_dump(mode="json") for t, p in self.profiles.items()},
            "state_before": before,
            "state_after": after,
        }
        result = decision.model_copy(
            update={
                "explanation": explanation,
                "duration_ms": (time.perf_counter_ns() - started) / 1e6,
            }
        )
        # Validate/freeze copied mappings before returning them to concurrent consumers.
        result = type(decision).model_validate(result.model_dump())
        result = result.model_copy(update={"duration_ms": (time.perf_counter_ns() - started) / 1e6})
        if self.logger is not None:
            self.logger.bind(candidate.trace_id, candidate.candidate_id).emit(
                module="scheduling",
                event_type="scheduler.decision",
                reason_code=result.reason_code,
                duration_ms=result.duration_ms,
                payload=result,
            )
        return result


class AuditedSelection:
    """Give the unchanged static baseline the same decision evidence as adaptive rules."""

    def __init__(self, base: TierSelectionPolicy, context: PolicyContext) -> None:
        self.base, self.context = base, context

    def select(self, candidate: Candidate, resources: ResourceSnapshot) -> SchedulingDecision:
        started = time.perf_counter_ns()
        result = self.base.select(candidate, resources)
        return self.context.record(
            "static_tier",
            candidate,
            resources,
            result,
            started,
            self.context.inputs(resources, self.context.clock()),
            {},
            {},
            {"adaptive_checks_enabled": False},
        )


class AuditedAdmission:
    def __init__(self, base: AdmissionPolicy, context: PolicyContext) -> None:
        self.base, self.context = base, context

    def decide(self, candidate: Candidate, resources: ResourceSnapshot) -> AdmissionDecision:
        started = time.perf_counter_ns()
        result = self.base.decide(candidate, resources)
        return self.context.record(
            "sequential",
            candidate,
            resources,
            result,
            started,
            self.context.inputs(resources, self.context.clock()),
            {},
            {},
            {"max_total_active": 1, "memory_checks_enabled": False},
        )


class ResourceQueueAwareTierPolicy:
    """Select only feasible tiers; cheap under pressure; upgrade after stable headroom."""

    def __init__(self, context: PolicyContext) -> None:
        self.context = context
        self.lock = threading.RLock()
        self.last: ModelTier | None = None
        self.low_since: int | None = None
        self.changed_at: int | None = None

    def state(self) -> dict[str, Any]:
        return {
            "last_tier": self.last.value if self.last else None,
            "low_since_ns": self.low_since,
            "changed_at_ns": self.changed_at,
        }

    def select(self, candidate: Candidate, resources: ResourceSnapshot) -> SchedulingDecision:
        started = time.perf_counter_ns()
        ctx, c = self.context, self.context.constraints
        with self.lock:
            now = ctx.clock()
            data, before = ctx.inputs(resources, now), self.state()
            free, cpu, gpu = data["free_bytes"], data["cpu_percent"], data["gpu_percent"]
            evaluated = {}
            feasible = []
            for tier in ctx.portfolio:
                p = ctx.profiles.get(tier)
                checks = {
                    "compatible_profile": p is not None,
                    "memory_fit": p is not None
                    and free is not None
                    and p.memory_bytes + c.memory_margin_bytes <= free,
                    "latency_fit": p is not None
                    and (
                        c.latency_slo_ms is None
                        or resources.oldest_wait_ms + p.latency_p95_ms <= c.latency_slo_ms
                    ),
                    "tier_capacity": data["active_jobs"].get(tier.value, 0) < c.per_tier_caps[tier],
                }
                evaluated[tier.value] = checks
                if all(checks.values()):
                    feasible.append(tier)
            reason = "NO_FEASIBLE_TIER"
            target = None
            escalation = None
            if feasible:
                cheapest = min(feasible, key=lambda t: (ctx.profiles[t].latency_ms, ORDER[t]))
                high_queue = resources.queue_length >= c.queue_high or (
                    c.queue_wait_slo_ms is not None
                    and resources.oldest_wait_ms >= c.queue_wait_slo_ms
                )
                low = (
                    data["fresh"]
                    and cpu is not None
                    and cpu <= c.cpu_low
                    and resources.queue_length <= c.queue_low
                    and (ctx.device == "cpu" or (gpu is not None and gpu <= c.gpu_low))
                    and bool(data["latency_ratios"])
                    and all(r < c.contention_ratio for r in data["latency_ratios"].values())
                    and not high_queue
                )
                target = cheapest
                if high_queue:
                    reason = "QUEUE_PRESSURE_HIGH"
                elif cpu is not None and cpu >= c.cpu_high:
                    reason = "CPU_PRESSURE_HIGH"
                elif ctx.device == "cuda" and gpu is not None and gpu >= c.gpu_high:
                    # A single GPU sample only prevents an upgrade; it never authorizes concurrency.
                    reason = "GPU_PRESSURE_HIGH"
                elif any(r >= c.contention_ratio for r in data["latency_ratios"].values()):
                    reason = "CONTENTION_DETECTED"
                elif not low:
                    reason = "INSUFFICIENT_HISTORY"
                else:
                    self.low_since = now if self.low_since is None else self.low_since
                    ready = now - self.low_since >= c.hysteresis_s * 1e9 and (
                        self.changed_at is None or now - self.changed_at >= c.cooldown_s * 1e9
                    )
                    stronger = sorted(feasible, key=ORDER.__getitem__)
                    if c.preserve_escalation_target and len(stronger) > 1:
                        escalation = stronger.pop()
                    desired = stronger[-1]
                    if ready:
                        target, reason = desired, "RESOURCES_AVAILABLE"
                    else:
                        target = self.last if self.last in feasible else cheapest
                        reason = "HYSTERESIS_HOLD"
                if not low:
                    self.low_since = None
            else:
                self.low_since = None
            if target != self.last:
                self.last, self.changed_at = target, now
            return ctx.record(
                "resource_queue_aware",
                candidate,
                resources,
                SchedulingDecision(
                    candidate_id=candidate.candidate_id, selected_tier=target, reason_code=reason
                ),
                started,
                data,
                before,
                self.state(),
                {
                    "tiers": evaluated,
                    "escalation_target": escalation.value if escalation else None,
                    "recall_is_reference_only": True,
                },
            )


class RoundRobinTierPolicy:
    """Diagnostic routing only; order is canonical across portfolio permutations."""

    def __init__(self, context: PolicyContext) -> None:
        self.context, self.index, self.lock = context, 0, threading.Lock()

    def select(self, candidate: Candidate, resources: ResourceSnapshot) -> SchedulingDecision:
        started = time.perf_counter_ns()
        with self.lock:
            before = self.index
            tier = self.context.portfolio[self.index]
            self.index = (self.index + 1) % len(self.context.portfolio)
            return self.context.record(
                "round_robin_diagnostic",
                candidate,
                resources,
                SchedulingDecision(
                    candidate_id=candidate.candidate_id,
                    selected_tier=tier,
                    reason_code="ROUND_ROBIN_DIAGNOSTIC",
                ),
                started,
                self.context.inputs(resources, self.context.clock()),
                {"index": before},
                {"index": self.index},
                {"diagnostic_only": True},
            )


class FixedConcurrencyAdmissionPolicy:
    """Atomic total/per-tier cap leases. This ablation deliberately omits memory rules."""

    def __init__(self, context: PolicyContext, capacity: int) -> None:
        if capacity < 1 or capacity > context.constraints.max_total_active:
            raise ValueError("invalid fixed capacity")
        self.context, self.capacity = context, capacity
        self.lock = threading.RLock()
        self.reservations: dict[str, tuple[ModelTier, int]] = {}
        self.serial = 0

    def state(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "reserved_jobs": len(self.reservations),
            "reserved_bytes": sum(v[1] for v in self.reservations.values()),
            "reserved_by_tier": {
                t.value: sum(v[0] == t for v in self.reservations.values())
                for t in self.context.portfolio
            },
            "serial": self.serial,
        }

    def decide(self, candidate: Candidate, resources: ResourceSnapshot) -> AdmissionDecision:
        if len(self.context.portfolio) != 1:
            raise ValueError(
                "tier-aware admission requires admit(candidate, selected_tier, snapshot)"
            )
        return self.admit(candidate, self.context.portfolio[0], resources)

    def _reason(self, tier: ModelTier, data: dict[str, Any]) -> str | None:
        # Count external observed jobs plus owned reservations. Double counting owned running
        # work is conservative, and avoids assuming sampled jobs are owned by this ledger.
        state = self.state()
        active = max(sum(data["active_jobs"].values()), data["active_jobs_total"] or 0)
        if active + state["reserved_jobs"] >= self.capacity:
            return "TOTAL_CAP_REACHED"
        if (
            data["active_jobs"].get(tier.value, 0) + state["reserved_by_tier"][tier.value]
            >= self.context.constraints.per_tier_caps[tier]
        ):
            return "PER_TIER_CAP_REACHED"
        return None

    def admit(
        self, candidate: Candidate, selected_tier: ModelTier, resources: ResourceSnapshot
    ) -> AdmissionDecision:
        started = time.perf_counter_ns()
        with self.lock:
            data = self.context.inputs(resources, self.context.clock())
            before = self.state()
            reason = (
                "NO_FEASIBLE_TIER"
                if selected_tier not in self.context.portfolio
                else self._reason(selected_tier, data)
            )
            return self._finish(
                candidate,
                selected_tier,
                resources,
                started,
                data,
                before,
                reason,
                0,
                "fixed_concurrency",
            )

    def _finish(
        self,
        candidate: Candidate,
        tier: ModelTier,
        resources: ResourceSnapshot,
        started: int,
        data: dict[str, Any],
        before: dict[str, Any],
        reason: str | None,
        memory: int,
        policy: str,
        success_reason: str = "RESOURCES_AVAILABLE",
    ) -> AdmissionDecision:
        token = None
        if reason is None:
            self.serial += 1
            token = f"reservation-{self.serial}"
            self.reservations[token] = (tier, memory)
        try:
            return self.context.record(
                policy,
                candidate,
                resources,
                AdmissionDecision(
                    candidate_id=candidate.candidate_id,
                    outcome="delayed" if reason else "admitted",
                    permitted_concurrency=self.capacity,
                    reason_code=reason or success_reason,
                    reservation_id=token,
                ),
                started,
                data,
                before,
                self.state(),
                {
                    "selected_tier": tier.value,
                    "requested_bytes": memory,
                    "memory_checks_enabled": policy in {"memory_contention_aware", "fixed_memory"},
                },
            )
        except BaseException:
            if token is not None:
                self.reservations.pop(token)
            raise

    def release(self, reservation_id: str, *, outcome: str = "completed") -> None:
        if outcome not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid reservation outcome")
        with self.lock:
            # Idempotence permits cleanup after a failure during completion reporting.
            self.reservations.pop(reservation_id, None)


class MemoryContentionAwareAdmissionPolicy(FixedConcurrencyAdmissionPolicy):
    def __init__(self, context: PolicyContext) -> None:
        super().__init__(context, 1)
        self.low_since: int | None = None
        self.changed_at: int | None = None
        self.gpu_high_since: int | None = None
        self.gpu_observation: int | None = None
        self.gpu_samples = 0

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "low_since_ns": self.low_since,
            "changed_at_ns": self.changed_at,
            "gpu_high_since_ns": self.gpu_high_since,
            "gpu_observation_ns": self.gpu_observation,
            "gpu_high_samples": self.gpu_samples,
        }

    def admit(
        self, candidate: Candidate, selected_tier: ModelTier, resources: ResourceSnapshot
    ) -> AdmissionDecision:
        started = time.perf_counter_ns()
        ctx, c = self.context, self.context.constraints
        with self.lock:
            now = ctx.clock()
            data, before = ctx.inputs(resources, now), self.state()
            p = ctx.profiles.get(selected_tier)
            cpu, gpu, free = data["cpu_percent"], data["gpu_percent"], data["free_bytes"]
            high_gpu = ctx.device == "cuda" and gpu is not None and gpu >= c.gpu_high
            if high_gpu:
                if resources.gpu_observed_monotonic_ns != self.gpu_observation:
                    self.gpu_samples += 1
                    self.gpu_observation = resources.gpu_observed_monotonic_ns
                    self.gpu_high_since = (
                        now if self.gpu_high_since is None else self.gpu_high_since
                    )
            else:
                self.gpu_high_since, self.gpu_observation, self.gpu_samples = None, None, 0
            sustained_gpu = (
                self.gpu_samples >= 2
                and self.gpu_high_since is not None
                and now - self.gpu_high_since >= c.hysteresis_s * 1e9
            )
            history = selected_tier.value in data["latency_ratios"]
            contention = any(v >= c.contention_ratio for v in data["latency_ratios"].values())
            reason = None
            if selected_tier not in ctx.portfolio or p is None:
                reason = "NO_FEASIBLE_TIER"
            elif free is None:
                reason = "INSUFFICIENT_HISTORY"
            elif free - before["reserved_bytes"] - p.memory_bytes < c.memory_margin_bytes:
                reason = "MEMORY_HEADROOM_LOW"
            elif contention:
                reason = "CONTENTION_DETECTED"
            elif cpu is not None and cpu >= c.cpu_high:
                reason = "CPU_PRESSURE_HIGH"
            elif sustained_gpu:
                reason = "GPU_PRESSURE_HIGH"
            low = (
                reason is None
                and history
                and cpu is not None
                and cpu <= c.cpu_low
                and resources.queue_length <= c.queue_low
                and (ctx.device == "cpu" or (gpu is not None and gpu <= c.gpu_low))
            )
            success_reason = "RESOURCES_AVAILABLE"
            if reason is not None:
                self.capacity, self.low_since, self.changed_at = 1, None, now
            elif low:
                self.low_since = now if self.low_since is None else self.low_since
                if (
                    now - self.low_since >= c.hysteresis_s * 1e9
                    and (self.changed_at is None or now - self.changed_at >= c.cooldown_s * 1e9)
                    and self.capacity < c.max_total_active
                ):
                    self.capacity += 1
                    self.changed_at, self.low_since = now, now
                elif self.capacity < c.max_total_active:
                    success_reason = "HYSTERESIS_HOLD"
            else:
                self.low_since = None
                if not history or cpu is None or (ctx.device == "cuda" and gpu is None) or high_gpu:
                    self.capacity = 1
                    success_reason = "INSUFFICIENT_HISTORY"
            if reason is None:
                reason = self._reason(selected_tier, data)
            return self._finish(
                candidate,
                selected_tier,
                resources,
                started,
                data,
                before,
                reason,
                p.memory_bytes if p else 0,
                "memory_contention_aware",
                success_reason,
            )

    def release(self, reservation_id: str, *, outcome: str = "completed") -> None:
        with self.lock:
            exists = reservation_id in self.reservations
            super().release(reservation_id, outcome=outcome)
            if exists and outcome == "failed":
                self.capacity, self.low_since, self.changed_at = 1, None, self.context.clock()
