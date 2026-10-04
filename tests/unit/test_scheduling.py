import io
import itertools
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from edge_triage.contracts import (
    Candidate,
    ModelTier,
    ResourceSnapshot,
    TierJobs,
    TierLatency,
    utc_now,
)
from edge_triage.event_logging import EventLogger
from edge_triage.scheduling.policies import (
    FixedConcurrencyAdmissionPolicy,
    MemoryContentionAwareAdmissionPolicy,
    PolicyContext,
    ResourceQueueAwareTierPolicy,
    RoundRobinTierPolicy,
)
from edge_triage.scheduling.profiles import SchedulerProfile
from edge_triage.scheduling.settings import SchedulerConstraints

T, M, L = ModelTier
PORTFOLIOS = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]


def candidate(i=0):
    return Candidate(
        run_id="r", trace_id=f"t{i}", candidate_id=f"c{i}", dataset_index=i, arrival_monotonic_ns=0
    )


def profiles():
    return {
        t: SchedulerProfile(
            tier=t,
            profile_id=t.value,
            checkpoint_id=t.value,
            latency_ms=i,
            latency_p95_ms=i * 1.2,
            memory_bytes=i * 100,
            memory_basis="synthetic_test",
        )
        for i, t in enumerate(ModelTier, 1)
    }


def snapshot(now=1_000_000_000, **updates):
    raw = dict(
        timestamp_utc=utc_now(),
        monotonic_ns=now,
        snapshot_id="s0",
        cpu_percent=10,
        available_ram_bytes=2000,
        gpu_memory_free_bytes=2000,
        gpu_utilization_percent=10,
        gpu_observed_monotonic_ns=now,
        queue_length=0,
        arrival_rate_per_s=None,
        recent_latency_by_tier=tuple(
            TierLatency(tier=t, mean_ms=i, samples=10) for i, t in enumerate(ModelTier, 1)
        ),
    )
    raw.update(updates)
    return ResourceSnapshot(**raw)


def context(
    portfolio=tuple(ModelTier),
    *,
    clock=lambda: 1_000_000_000,
    device="cpu",
    logger=None,
    **constraints,
):
    c = dict(
        memory_margin_bytes=50,
        hysteresis_s=0,
        cooldown_s=0,
        preserve_escalation_target=False,
        per_tier_caps={t: 4 for t in ModelTier},
    )
    c.update(constraints)
    return PolicyContext(
        portfolio, profiles(), SchedulerConstraints(**c), clock=clock, device=device, logger=logger
    )


@pytest.mark.parametrize(
    "portfolio,signals,expected,reason",
    [
        ((T, M, L), {}, L, "RESOURCES_AVAILABLE"),
        ((T, M, L), {"queue_length": 8}, T, "QUEUE_PRESSURE_HIGH"),
        ((T, M, L), {"available_ram_bytes": 160}, T, "RESOURCES_AVAILABLE"),
        ((T, M, L), {"available_ram_bytes": 149}, None, "NO_FEASIBLE_TIER"),
        ((T, M), {}, M, "RESOURCES_AVAILABLE"),
        ((L,), {}, L, "RESOURCES_AVAILABLE"),
        ((T, L), {"active_jobs_by_tier": (TierJobs(tier=L, count=4),)}, T, "RESOURCES_AVAILABLE"),
        ((T, M, L), {"cpu_percent": 95}, T, "CPU_PRESSURE_HIGH"),
        ((T, M, L), {"cpu_percent": None}, T, "INSUFFICIENT_HISTORY"),
        ((T, M, L), {"available_ram_bytes": None}, None, "NO_FEASIBLE_TIER"),
        ((T, M, L), {"recent_latency_by_tier": ()}, T, "INSUFFICIENT_HISTORY"),
    ],
)
def test_selection_scenarios(portfolio, signals, expected, reason):
    result = ResourceQueueAwareTierPolicy(context(portfolio)).select(
        candidate(), snapshot(**signals)
    )
    assert (result.selected_tier, result.reason_code) == (expected, reason)


@pytest.mark.parametrize("portfolio", PORTFOLIOS)
def test_portfolio_legality_property(portfolio):
    # Exhaustive small Cartesian domain, including missing/stale signals and tight budgets.
    for memory, cpu, queue in itertools.product(
        (None, 0, 160, 260, 1000), (None, 10, 95), (0, 3, 9)
    ):
        policy = ResourceQueueAwareTierPolicy(context(portfolio))
        result = policy.select(
            candidate(), snapshot(available_ram_bytes=memory, cpu_percent=cpu, queue_length=queue)
        )
        assert result.selected_tier is None or result.selected_tier in portfolio
        if result.selected_tier:
            assert profiles()[result.selected_tier].memory_bytes + 50 <= memory


@pytest.mark.parametrize(
    "signals,tier,reason",
    [
        ({}, T, "RESOURCES_AVAILABLE"),
        ({"available_ram_bytes": 149}, T, "MEMORY_HEADROOM_LOW"),
        ({"active_jobs_total": 4}, T, "TOTAL_CAP_REACHED"),
        ({"active_jobs_by_tier": (TierJobs(tier=L, count=1),)}, L, "PER_TIER_CAP_REACHED"),
        (
            {"recent_latency_by_tier": (TierLatency(tier=T, mean_ms=3, samples=10),)},
            T,
            "CONTENTION_DETECTED",
        ),
        ({"cpu_percent": 95}, T, "CPU_PRESSURE_HIGH"),
        ({"available_ram_bytes": None}, T, "INSUFFICIENT_HISTORY"),
    ],
)
def test_admission_scenarios(signals, tier, reason):
    policy = MemoryContentionAwareAdmissionPolicy(context(per_tier_caps={T: 4, M: 2, L: 1}))
    result = policy.admit(candidate(), tier, snapshot(**signals))
    assert result.reason_code == reason
    assert result.outcome == ("admitted" if reason == "RESOURCES_AVAILABLE" else "delayed")
    if result.reservation_id:
        policy.release(result.reservation_id)
    assert policy.state()["reserved_bytes"] == 0


def test_gpu_missing_stale_and_single_observation_never_increases_concurrency():
    now = [1_000_000_000]
    policy = MemoryContentionAwareAdmissionPolicy(context(device="cuda", clock=lambda: now[0]))
    for fields in ({"gpu_observed_monotonic_ns": None}, {"gpu_memory_free_bytes": None}):
        result = policy.admit(candidate(), T, snapshot(**fields))
        assert (result.outcome, result.reason_code) == ("delayed", "INSUFFICIENT_HISTORY")
    high = snapshot(gpu_utilization_percent=99, recent_latency_by_tier=())
    for _ in range(3):
        result = policy.admit(candidate(), T, high)
        assert result.permitted_concurrency == 1 and result.reason_code == "INSUFFICIENT_HISTORY"
        policy.release(result.reservation_id)
    now[0] += 100_000_000
    result = policy.admit(candidate(), T, snapshot(now[0], gpu_utilization_percent=99))
    assert result.reason_code == "GPU_PRESSURE_HIGH" and result.outcome == "delayed"
    now[0] += 3_000_000_000
    result = policy.admit(candidate(), T, high)
    assert result.reason_code == "INSUFFICIENT_HISTORY" and result.outcome == "delayed"
    # CPU execution ignores unavailable GPU measurements.
    cpu = MemoryContentionAwareAdmissionPolicy(context())
    assert cpu.admit(candidate(), T, snapshot(gpu_utilization_percent=None)).outcome == "admitted"


def test_hysteresis_cooldown_and_failure_decrease():
    now = [1_000_000_000]
    ctx = context(clock=lambda: now[0], hysteresis_s=1, cooldown_s=2)
    selector, admission = (
        ResourceQueueAwareTierPolicy(ctx),
        MemoryContentionAwareAdmissionPolicy(ctx),
    )

    def step():
        s = selector.select(candidate(), snapshot(now[0]))
        a = admission.admit(candidate(), T, snapshot(now[0]))
        admission.release(a.reservation_id)
        return s, a

    s, a = step()
    assert s.selected_tier == T and s.reason_code == a.reason_code == "HYSTERESIS_HOLD"
    now[0] += 1_000_000_000
    s, a = step()
    assert s.selected_tier == T and a.permitted_concurrency == 2
    now[0] += 1_000_000_000
    s, a = step()
    assert s.selected_tier == L and a.permitted_concurrency == 2
    failed = admission.admit(candidate(), T, snapshot(now[0]))
    admission.release(failed.reservation_id, outcome="failed")
    assert admission.capacity == 1
    now[0] += 1_000_000_000
    assert step()[1].permitted_concurrency == 1
    now[0] += 1_000_000_000
    assert step()[1].permitted_concurrency == 2


def test_memory_race_and_release_conservation():
    policy = MemoryContentionAwareAdmissionPolicy(context(max_total_active=4))
    policy.capacity = 4
    # High queue suppresses growth but does not reduce this warmed cap.
    snap = snapshot(available_ram_bytes=250, queue_length=8)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda i: policy.admit(candidate(i), T, snap), range(40)))
    admitted = [r for r in results if r.outcome == "admitted"]
    assert len(admitted) == 2
    assert policy.state()["reserved_jobs"] == 2 and policy.state()["reserved_bytes"] == 200
    assert {r.reason_code for r in results if r.outcome == "delayed"} == {"MEMORY_HEADROOM_LOW"}
    for i, result in enumerate(admitted):
        policy.release(result.reservation_id, outcome=("completed", "cancelled")[i])
        policy.release(result.reservation_id)
    assert policy.state()["reserved_bytes"] == policy.state()["reserved_jobs"] == 0
    assert policy.admit(candidate(99), T, snap).outcome == "admitted"


def test_fixed_cap_race_and_round_robin():
    ctx = context((T, L))
    policy = FixedConcurrencyAdmissionPolicy(ctx, 3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: policy.admit(candidate(i), T, snapshot()), range(20)))
    assert sum(r.outcome == "admitted" for r in results) == 3
    rr = RoundRobinTierPolicy(ctx)
    assert [rr.select(candidate(i), snapshot()).selected_tier for i in range(4)] == [T, L, T, L]


def test_determinism_explanations_and_logs(tmp_path):
    logger = EventLogger(tmp_path / "events.jsonl", "r", "0" * 64, console=io.StringIO())
    policies = [ResourceQueueAwareTierPolicy(context(logger=logger)) for _ in range(2)]
    snap = snapshot()
    results = [p.select(candidate(), snap) for p in policies]
    assert results[0].model_dump(exclude={"duration_ms"}) == results[1].model_dump(
        exclude={"duration_ms"}
    )
    admission = MemoryContentionAwareAdmissionPolicy(context(logger=logger))
    admission.admit(candidate(), T, snap)
    logger.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 3
    for event in events:
        assert event["duration_ms"] >= 0
        detail = event["payload"]["explanation"]
        assert detail["snapshot_id"] == "s0" and detail["candidate_id"] == "c0"
        assert all(
            k in detail
            for k in (
                "policy",
                "version",
                "portfolio",
                "constraints",
                "profiles",
                "state_before",
                "state_after",
                "evaluated",
                "inputs",
            )
        )
        assert "TRUNCATED" not in json.dumps(event)
        assert "label" not in json.dumps(detail)


def test_constraints_latency_slo_escalation_and_missing_profile():
    with pytest.raises(ValidationError):
        SchedulerConstraints(cpu_low=95)
    with pytest.raises(ValidationError):
        SchedulerConstraints(per_tier_caps={T: 0})
    ctx = context(latency_slo_ms=2, preserve_escalation_target=True)
    result = ResourceQueueAwareTierPolicy(ctx).select(candidate(), snapshot())
    assert result.selected_tier == T
    result = ResourceQueueAwareTierPolicy(context(preserve_escalation_target=True)).select(
        candidate(), snapshot()
    )
    assert (
        result.selected_tier == M
        and result.explanation["evaluated"]["escalation_target"] == "large"
    )
    ctx = PolicyContext(
        (T, L), {T: profiles()[T]}, context().constraints, clock=lambda: 1_000_000_000
    )
    assert ResourceQueueAwareTierPolicy(ctx).select(candidate(), snapshot()).selected_tier == T


def test_scheduler_config_and_nested_decision_pickle_preserve_immutability():
    from edge_triage.config import AppConfig

    config = AppConfig()
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config and restored.config_hash == config.config_hash
    with pytest.raises(TypeError, match="immutable"):
        restored.scheduler.constraints.per_tier_caps[T] = 9
    decision = ResourceQueueAwareTierPolicy(context()).select(candidate(), snapshot())
    restored_decision = pickle.loads(pickle.dumps(decision))
    assert restored_decision == decision
    with pytest.raises(TypeError, match="immutable"):
        restored_decision.explanation["portfolio"].append("invalid")
