import itertools
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import CandidateEvent, ModelTier
from edge_triage.runtime import streaming
from edge_triage.scheduling.profiles import scheduler_profiles

ROOT = Path(__file__).resolve().parents[2]
PORTFOLIOS = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]


class AvailableCPU:
    def collect(self):
        return {"cpu_percent": 10, "available_ram_bytes": 16_000_000_000}


class UnavailableGPU:
    def __init__(self, *args):
        pass

    def collect(self):
        return {"availability": {"gpu": "unavailable:test"}}


@pytest.mark.parametrize("portfolio", PORTFOLIOS)
def test_adaptive_sequential_all_portfolios_frozen_trace(tmp_path, monkeypatch, portfolio):
    monkeypatch.setattr(streaming, "HostCollector", AvailableCPU)
    monkeypatch.setattr(streaming, "NvidiaCollector", UnavailableGPU)
    raw = load_config(ROOT / "configs/experiments/module07_v1.yaml").model_dump()
    raw["paths"].update(runs_dir=tmp_path / "runs", artifacts_dir=tmp_path / "artifacts")
    raw["models"]["tier_portfolio"] = portfolio
    raw["monitor"]["interval_s"] = 0.01
    raw["workload"]["trace"] = ROOT / "artifacts/traces/backlog_periodic_ffee5a44e7aa7e08.json"
    # Read the original hash-checked trace, then use a short prefix in this isolated test.
    original = streaming.load_trace
    monkeypatch.setattr(streaming, "load_trace", lambda *args: original(*args)[:8])
    directory = streaming.run_adaptive_sequential(AppConfig.model_validate(raw))
    summary = json.loads((directory / "metrics/adaptive_sequential_summary.json").read_text())
    assert (summary["completed"], summary["failed"]) == (8, 0)
    rows = pq.read_table(directory / "candidates.parquet").to_pylist()
    assert all(r["selected_tiers"] in {t.value for t in portfolio} for r in rows)
    assert all(r["concurrency"] == 1 for r in rows)
    assert summary["test_evaluated"] is False
    decisions = [
        CandidateEvent.model_validate_json(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "scheduler.decision"
    ]
    assert len(decisions) == 16
    assert all(
        d.payload["explanation"]["portfolio"] == [t.value for t in portfolio] for d in decisions
    )


@pytest.mark.parametrize("failure", [False, True])
def test_real_inference_reservations_released_with_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(streaming, "HostCollector", AvailableCPU)
    monkeypatch.setattr(streaming, "NvidiaCollector", UnavailableGPU)
    raw = load_config(ROOT / "configs/experiments/module07_v1.yaml").model_dump()
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    raw["admission"]["policy"] = "memory_contention_aware"
    raw["scheduler"]["constraints"]["max_total_active"] = 1
    # Exercise the initial single-job safe path, without pretending this measures contention.
    raw["scheduler"]["constraints"]["min_latency_samples"] = 100
    original_trace, original_assemble = streaming.load_trace, streaming.assemble
    monkeypatch.setattr(streaming, "load_trace", lambda *args: original_trace(*args)[:3])
    parts_used = []

    class FailOne:
        def __init__(self, base):
            self.base = base

        def predict(self, item, tier):
            if failure and item.candidate.candidate_id.endswith("occurrence-1"):
                raise RuntimeError("test inference failure")
            return self.base.predict(item, tier)

    def assemble(*args):
        parts = original_assemble(*args)
        parts.models = {tier: FailOne(model) for tier, model in parts.models.items()}
        parts_used.append(parts.admission)
        return parts

    monkeypatch.setattr(streaming, "assemble", assemble)
    directory = streaming.run_adaptive_sequential(AppConfig.model_validate(raw))
    summary = json.loads((directory / "metrics/adaptive_sequential_summary.json").read_text())
    assert (summary["completed"], summary["failed"]) == (3 - int(failure), int(failure))
    assert parts_used[0].state()["reserved_jobs"] == parts_used[0].state()["reserved_bytes"] == 0


def test_frozen_profile_compatibility_rejections():
    config = load_config(ROOT / "configs/experiments/module07_v1.yaml")
    rows = scheduler_profiles(config)
    assert set(rows) == set(ModelTier)
    assert rows[ModelTier.TINY].latency_ms < rows[ModelTier.LARGE].latency_ms
    for change in (
        {"profiles_sha256": "0" * 64},
        {"hardware_id": "other-hardware"},
        {"device": "cuda"},
    ):
        altered = config.model_copy(
            update={"scheduler": config.scheduler.model_copy(update=change)}
        )
        with pytest.raises(ValueError):
            scheduler_profiles(altered)


def test_live_counters_override_stale_busy_snapshot_for_lease_admission():
    from edge_triage.contracts import TierJobs
    from edge_triage.monitoring.estimators import RuntimeCounters
    from edge_triage.scheduling.policies import FixedConcurrencyAdmissionPolicy
    from tests.unit.test_scheduling import candidate, context, snapshot

    policy = FixedConcurrencyAdmissionPolicy(context(), 1)
    observed = streaming.ObservedAdmission(policy, RuntimeCounters(2, 100))
    result = observed.admit(
        candidate(),
        ModelTier.TINY,
        snapshot(
            now=time.monotonic_ns(),
            active_jobs_total=1,
            active_jobs_by_tier=(TierJobs(tier=ModelTier.TINY, count=1),),
        ),
    )
    assert result.outcome == "admitted"
    observed.release(result.reservation_id)
    assert policy.state()["reserved_jobs"] == 0
