import itertools
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from edge_triage.config import load_config
from edge_triage.contracts import AdmissionDecision, Candidate, EscalationDecision, ModelTier
from edge_triage.event_logging import EventLogger
from edge_triage.health import hash_file
from edge_triage.runtime.components import LoggerEventSink, SequentialExecutor
from edge_triage.runtime.factory import EXECUTORS, MODELS, assemble, run_smoke
from edge_triage.runtime.interfaces import CandidateInput
from edge_triage.runtime.pipeline import Pipeline
from edge_triage.runtime.records import CandidateRecord, ParquetRecordWriter
from edge_triage.runtime.reference import StaticTierPolicy
from edge_triage.runtime.state import PipelineState, StateMachine
from edge_triage.uncertainty.policies import CascadeEscalationPolicy
from edge_triage.uncertainty.sampling import SamplingError, summarize
from tests.fixtures.dummy import DummyModelAdapter

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def smoke_config(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/experiments/smoke.yaml")
    raw = config.model_dump()
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    raw["pipeline"]["model_adapter"] = "dummy"
    monkeypatch.setitem(MODELS, "dummy", lambda config: DummyModelAdapter())
    return type(config).model_validate(raw)


def source(count=3):
    return [
        CandidateInput(
            Candidate(
                run_id="test-run",
                trace_id=f"trace-{index}",
                candidate_id=f"c{index}",
                dataset_index=index,
                arrival_monotonic_ns=0,
            ),
            torch.full((3, 30, 30), 12345.0),
            index % 2,
            120000,
        )
        for index in range(count)
    ]


def execute(config, tmp_path, components=None, count=3):
    parts = components or assemble(config)
    logger = EventLogger(tmp_path / "events.jsonl", "test-run", config.config_hash)
    writer = ParquetRecordWriter(tmp_path / "candidates.parquet", batch_size=2)
    pipeline = Pipeline(config, parts, logger, writer)
    try:
        totals = pipeline.run(source(count))
    finally:
        logger.close()
    records = pq.read_table(tmp_path / "candidates.parquet").to_pylist()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    return totals, records, events, writer, parts


def test_end_to_end_records_timestamps_and_no_tensor_serialization(smoke_config, tmp_path):
    totals, records, events, writer, parts = execute(smoke_config, tmp_path)
    assert totals == (3, 0) and writer.closed and parts.executor.closed
    assert len(records) == 3  # Flushes the final partial batch of one.
    table = pq.read_table(tmp_path / "candidates.parquet")
    assert table.schema.metadata[b"purpose"] == b"NON_SCIENTIFIC_SMOKE_TEST"
    assert all(
        not pa.types.is_binary(field.type) and not pa.types.is_list(field.type)
        for field in table.schema
    )
    for record in records:
        CandidateRecord.model_validate(record)
        assert (
            record["arrival_ns"]
            <= record["queue_ns"]
            <= record["start_ns"]
            <= record["end_ns"]
            <= record["final_ns"]
        )
        assert record["queue_ms"] == (record["start_ns"] - record["queue_ns"]) / 1e6
        assert record["end_to_end_ms"] == (record["final_ns"] - record["arrival_ns"]) / 1e6
        assert record["inference_ms"] == (record["end_ns"] - record["start_ns"]) / 1e6
        assert record["input_bytes"] == record["output_bytes"] == 120000
        assert "image" not in record and "tensor" not in record
        correlated = [event for event in events if event["candidate_id"] == record["candidate_id"]]
        assert all(
            event["trace_id"] == record["trace_id"]
            and event["run_id"] == "test-run"
            and event["reason_code"]
            for event in correlated
        )
        assert {event["event_type"] for event in correlated} >= {
            "data.arrived",
            "scheduler.queued",
            "scheduler.selected",
            "scheduler.admission",
            "inference.predicted",
            "uncertainty.evaluated",
            "uncertainty.escalation",
            "triage.threshold",
            "triage.decided",
            "inference.candidate_final",
        }
    assert '"image"' not in json.dumps(events) and '"tensor"' not in json.dumps(events)


def test_candidate_failure_isolation_and_flush(smoke_config, tmp_path):
    class FailingModel(DummyModelAdapter):
        def predict(self, item, tier):
            if item.candidate.candidate_id == "c1":
                raise ValueError("intentional model failure")
            return super().predict(item, tier)

    parts = assemble(smoke_config)
    parts.models[ModelTier.TINY] = FailingModel()
    totals, records, events, _, _ = execute(smoke_config, tmp_path, parts)
    assert totals == (2, 1)
    assert [record["status"] for record in records] == ["completed", "failed", "completed"]
    assert records[1]["error_code"] == "ValueError"
    assert any(event["event_type"] == "error.candidate_failed" for event in events)


def test_policy_and_executor_substitution(smoke_config, tmp_path, monkeypatch):
    class RecordingExecutor(SequentialExecutor):
        calls = 0

        def execute(self, model, item, tier):
            self.calls += 1
            return super().execute(model, item, tier)

    monkeypatch.setitem(EXECUTORS, "sequential", lambda config: RecordingExecutor())
    parts = assemble(smoke_config)
    parts.selector = StaticTierPolicy(ModelTier.LARGE)
    totals, records, _, _, _ = execute(smoke_config, tmp_path, parts)
    assert totals == (3, 0) and parts.executor.calls == 3
    assert all(record["selected_tiers"] == "large" for record in records)


def test_admission_delay_sleeps_and_timeout_is_bounded(smoke_config, tmp_path):
    class DelayOnce:
        calls = 0

        def decide(self, candidate, resources):
            self.calls += 1
            return AdmissionDecision(
                candidate_id=candidate.candidate_id,
                outcome="delayed" if self.calls == 1 else "admitted",
                permitted_concurrency=1,
                reason_code="TEST_DELAY",
            )

    parts = assemble(smoke_config)
    parts.admission = DelayOnce()
    logger = EventLogger(tmp_path / "events.jsonl", "test-run", smoke_config.config_hash)
    writer = ParquetRecordWriter(tmp_path / "candidates.parquet")
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        time.sleep(delay)

    try:
        assert Pipeline(smoke_config, parts, logger, writer, sleep=sleep).run(source(1)) == (1, 0)
    finally:
        logger.close()
    assert sleeps == [smoke_config.pipeline.admission_retry_s]
    assert parts.admission.calls == 2


def test_perpetual_delay_fails_candidate_without_busy_spin(smoke_config, tmp_path):
    class DelayForever:
        calls = 0

        def decide(self, candidate, resources):
            self.calls += 1
            return AdmissionDecision(
                candidate_id=candidate.candidate_id,
                outcome="delayed",
                permitted_concurrency=1,
                reason_code="WAIT",
            )

    raw = smoke_config.model_dump()
    raw["pipeline"]["admission_timeout_s"] = 0.025
    config = type(smoke_config).model_validate(raw)
    parts = assemble(config)
    parts.admission = DelayForever()
    totals, records, _, _, _ = execute(config, tmp_path, parts, count=1)
    assert totals == (0, 1) and parts.admission.calls <= 5
    assert records[0]["error_code"] == "AdmissionTimeoutError"
    assert records[0]["start_ns"] is None


def test_escalation_hook_reenters_admission(smoke_config, tmp_path):
    class EscalateOnce:
        def decide(self, prediction, uncertainty):
            escalate = prediction.tier == ModelTier.TINY
            return EscalationDecision(
                candidate_id=prediction.candidate_id,
                from_tier=prediction.tier,
                escalate=escalate,
                to_tier=ModelTier.MEDIUM if escalate else None,
                reason_code="TEST_ESCALATION",
            )

    parts = assemble(smoke_config)
    parts.escalation = EscalateOnce()
    totals, records, events, _, _ = execute(smoke_config, tmp_path, parts, count=1)
    assert totals == (1, 0) and records[0]["escalation_path"] == "tiny->medium"
    assert sum(event["event_type"] == "scheduler.admission" for event in events) == 2


@pytest.mark.parametrize(
    "portfolio", [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]
)
def test_complete_cascade_chains_cost_and_leases(smoke_config, tmp_path, portfolio):
    class Unresolved:
        def evaluate(self, prediction):
            return summarize(prediction, [0, 1, 0, 1, 0], 0).model_copy(update={"unresolved": True})

    class Leases:
        active = None
        releases = []

        def decide(self, candidate, snapshot):
            raise AssertionError("tier admission required")

        def admit(self, candidate, tier, snapshot):
            assert self.active is None
            self.active = tier.value
            return AdmissionDecision(
                candidate_id=candidate.candidate_id,
                outcome="admitted",
                permitted_concurrency=1,
                reason_code="TEST_LEASE",
                reservation_id=self.active,
            )

        def release(self, reservation_id, *, outcome="completed"):
            assert self.active == reservation_id
            self.releases.append(outcome)
            self.active = None

    parts = assemble(smoke_config)
    parts.models = {t: parts.models[t] for t in portfolio}
    parts.selector = StaticTierPolicy(portfolio[0])
    parts.uncertainty = Unresolved()
    parts.escalation = CascadeEscalationPolicy(portfolio)
    parts.admission = Leases()
    totals, rows, events, _, _ = execute(smoke_config, tmp_path, parts, count=1)
    assert totals == (1, 0)
    row = rows[0]
    assert row["selected_tiers"] == ",".join(t.value for t in portfolio)
    assert row["additional_forward_passes"] == 5 * len(portfolio)
    assert row["conservative_fallback"] == 1
    assert row["final_trusted_tier"] == portfolio[-1].value
    assert row["uq_ms"] > 0 and row["escalation_decision_ms"] > 0
    assert parts.admission.active is None
    assert parts.admission.releases == ["completed"] * len(portfolio)
    final = next(e for e in events if e["event_type"] == "inference.candidate_final")
    assert "_truncated" not in json.dumps(final) and "TRUNCATED" not in json.dumps(final)
    references = final["payload"]["resource_snapshot_refs"]
    snapshots = json.loads(row["resource_snapshot_json"])
    assert len(references) == len(snapshots) == len(portfolio)
    assert [r["monotonic_ns"] for r in references] == [s["monotonic_ns"] for s in snapshots]
    assert "resource_snapshot_json" not in final["payload"]
    for name, event_type in [
        ("prediction", "inference.predicted"),
        ("uncertainty", "uncertainty.evaluated"),
        ("escalation", "uncertainty.escalation"),
    ]:
        chain = json.loads(row[f"{name}_chain_json"])
        assert len(chain) == len(portfolio)
        assert chain == [e["payload"] for e in events if e["event_type"] == event_type]
        assert final["payload"][f"{name}_chain"] == chain


def test_failed_sampling_accounts_attempts_and_releases(smoke_config, tmp_path):
    class Broken:
        def evaluate(self, prediction):
            if prediction.candidate_id == "c1":
                raise SamplingError(3, 2, 0.01)
            return summarize(prediction, [0, 1], 0)

    from edge_triage.scheduling.policies import FixedConcurrencyAdmissionPolicy, PolicyContext
    from edge_triage.scheduling.profiles import scheduler_profiles

    real = load_config(ROOT / "configs/experiments/module07_v1.yaml")
    parts = assemble(smoke_config)
    parts.admission = FixedConcurrencyAdmissionPolicy(
        PolicyContext(
            real.models.tier_portfolio,
            scheduler_profiles(real),
            real.scheduler.constraints,
            device="cpu",
        ),
        1,
    )
    parts.uncertainty = Broken()
    totals, rows, events, _, _ = execute(smoke_config, tmp_path, parts)
    assert totals == (2, 1)
    assert rows[1]["additional_forward_passes"] == 3
    assert rows[1]["failed_uq_attempts"] == 3 and rows[1]["failed_uq_completed"] == 2
    assert rows[1]["uq_ms"] > 0
    assert any(e["event_type"] == "uncertainty.failed" for e in events)
    # The next candidate completing under cap1 proves the failed lease was released.
    assert rows[2]["status"] == "completed"


@pytest.mark.parametrize("invalid", ["cycle", "disabled", "maximum"])
def test_pipeline_guards_untrusted_escalation_hooks(smoke_config, tmp_path, invalid):
    class BadRoute:
        def decide(self, prediction, uncertainty):
            # model_copy simulates a substitute policy bypassing contract validation.
            return EscalationDecision(
                candidate_id=prediction.candidate_id,
                from_tier=prediction.tier,
                escalate=False,
                reason_code="TEST_BAD_ROUTE",
            ).model_copy(
                update={
                    "escalate": True,
                    "to_tier": ModelTier.TINY if invalid == "cycle" else ModelTier.LARGE,
                }
            )

    raw = smoke_config.model_dump()
    if invalid == "maximum":
        raw["escalation"]["max_escalations"] = 0
    config = type(smoke_config).model_validate(raw)
    parts = assemble(config)
    if invalid == "disabled":
        parts.models.pop(ModelTier.LARGE)
    parts.escalation = BadRoute()
    totals, rows, _, _, _ = execute(config, tmp_path, parts, count=1)
    assert totals == (0, 1)
    assert rows[0]["error_code"] == "ValueError"
    assert len(json.loads(rows[0]["prediction_chain_json"])) == 1


def test_uq_cancellation_releases_reservation_and_closes(smoke_config, tmp_path):
    class CancelledUQ:
        def evaluate(self, prediction):
            raise KeyboardInterrupt("diagnostic cancellation")

    class Lease:
        outcome = None

        def decide(self, candidate, resources):
            raise AssertionError("tier interface expected")

        def admit(self, candidate, tier, resources):
            return AdmissionDecision(
                candidate_id=candidate.candidate_id,
                outcome="admitted",
                permitted_concurrency=1,
                reason_code="TEST_LEASE",
                reservation_id="lease",
            )

        def release(self, reservation_id, *, outcome="completed"):
            assert reservation_id == "lease"
            self.outcome = outcome

    parts = assemble(smoke_config)
    parts.uncertainty, parts.admission = CancelledUQ(), Lease()
    with pytest.raises(KeyboardInterrupt):
        execute(smoke_config, tmp_path, parts, count=1)
    assert parts.admission.outcome == "cancelled" and parts.executor.closed
    assert pq.read_table(tmp_path / "candidates.parquet").num_rows == 0


def test_typed_event_bridge_preserves_clocks_and_redacts(smoke_config, tmp_path):
    logger = EventLogger(tmp_path / "events.jsonl", "test-run", smoke_config.config_hash)
    sink = LoggerEventSink(logger)
    captured = []

    class Capture:
        def emit(self, event):
            captured.append(event)
            sink.emit(event)

    machine = StateMachine(source(1)[0].candidate, smoke_config.config_hash, Capture())
    machine.transition(PipelineState.QUEUED)
    logger.close()
    actual = json.loads((tmp_path / "events.jsonl").read_text())
    assert actual == captured[0].model_dump(mode="json")


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs/experiments").glob("smoke*.yaml")), ids=lambda p: p.stem
)
def test_all_seven_portfolios_real_data_smoke(path, tmp_path):
    config = load_config(path)
    raw = config.model_dump()
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    directory = run_smoke(type(config).model_validate(raw), limit=2)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["purpose"] == "NON_SCIENTIFIC_SMOKE_TEST"
    assert (
        manifest["split_hash"]["digest"]
        == "fbdce973bc280ab81bd262f49fd0492f9b03e590b3ebbac977cbe2eb2a2f2d8b"
    )
    assert manifest["preprocessing_hash"] is not None
    records = pq.read_table(directory / "candidates.parquet").to_pylist()
    assert len(records) == 2 and all(record["status"] == "completed" for record in records)
    assert all(
        record["selected_tiers"] == config.models.tier_portfolio[0].value for record in records
    )
    assert records[0]["threshold_artifact_sha256"] == hash_file(
        directory / "artifacts/common_threshold.txt"
    )
    assert manifest["threshold_hashes"][0]["digest"] == records[0]["threshold_artifact_sha256"]
    resolved = load_config(directory / "config.resolved.yaml")
    assert manifest["config_hash"] == resolved.config_hash


def test_refuses_scientific_dummy_and_unsupported_backend(smoke_config):
    raw = smoke_config.model_dump()
    raw["experiment"]["purpose"] = "science"
    with pytest.raises(ValueError, match="NON_SCIENTIFIC"):
        assemble(type(smoke_config).model_validate(raw))
    raw = smoke_config.model_dump()
    raw["admission"]["execution_mode"] = "dynamic_concurrency"
    with pytest.raises(ValueError, match="not implemented"):
        assemble(type(smoke_config).model_validate(raw))
