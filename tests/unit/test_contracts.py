import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from edge_triage.contracts import (
    BOGUS_LABEL,
    REAL_LABEL,
    AdmissionDecision,
    ArtifactHash,
    Candidate,
    CandidateEvent,
    CandidateState,
    EscalationDecision,
    ModelProfile,
    ModelTier,
    PredictionResult,
    ResourceSnapshot,
    RunManifest,
    SchedulingDecision,
    TriageDecision,
    UncertaintyResult,
    p_real_from_logit,
    utc_now,
)


def prediction(**changes):
    return PredictionResult.model_validate(
        {
            "candidate_id": "c1",
            "tier": "tiny",
            "checkpoint_id": "ck1",
            "logit": 0,
            "p_real": 0.5,
            "inference_ms": 0.25,
            **changes,
        }
    )


def candidate():
    return Candidate(
        run_id="r1", trace_id="t1", candidate_id="c1", dataset_index=0, arrival_monotonic_ns=1
    )


def examples():
    now = utc_now()
    return [
        candidate(),
        prediction(),
        UncertaintyResult(
            candidate_id="c1",
            tier=ModelTier.TINY,
            method="mc_dropout",
            passes=10,
            mean_p_real=0.6,
            variance=0.01,
            predictive_entropy=0.6,
            duration_ms=2,
        ),
        ResourceSnapshot(
            timestamp_utc=now,
            monotonic_ns=1,
            cpu_percent=20,
            available_ram_bytes=1024,
            queue_length=1,
            arrival_rate_per_s=3,
        ),
        SchedulingDecision(candidate_id="c1", selected_tier=ModelTier.TINY, reason_code="FIXED"),
        AdmissionDecision(
            candidate_id="c1", outcome="admitted", permitted_concurrency=1, reason_code="CAPACITY"
        ),
        EscalationDecision(
            candidate_id="c1",
            from_tier=ModelTier.TINY,
            to_tier=ModelTier.LARGE,
            escalate=True,
            reason_code="UNCERTAIN",
        ),
        TriageDecision(
            candidate_id="c1",
            action="transmit",
            threshold_artifact_id="th1",
            threshold_artifact_sha256="a" * 64,
            p_real=0.9,
            reason_code="SURVIVAL",
            input_bytes=120,
            output_bytes=120,
        ),
        CandidateEvent(
            timestamp_utc=now,
            monotonic_ns=1,
            level="INFO",
            run_id="r1",
            trace_id="t1",
            candidate_id="c1",
            module="tests",
            event_type="run.started",
            config_hash="a" * 64,
            message="example",
            reason_code="TEST",
            duration_ms=0,
            payload={"nested": {"value": 1}},
            error_type=None,
            error_message=None,
        ),
        RunManifest(
            run_id="r1",
            purpose="test",
            started_utc=now,
            config_hash="a" * 64,
            dataset_hashes=(),
            git_commit=None,
            git_dirty=None,
            environment={"dependencies": {"test": "1"}, "gpus": [{"name": "test"}]},
            seeds={"test": 1},
            provenance_ids=("engineering-foundation",),
        ),
        ModelProfile(
            profile_id="p1",
            tier=ModelTier.MEDIUM,
            checkpoint_id="ck1",
            device="cpu",
            batch_size=1,
            concurrency=1,
            input_shape=(3, 30, 30),
            latency_p50_ms=1,
            latency_p95_ms=2,
            peak_memory_bytes=100,
            samples=100,
            measured_utc=now,
        ),
        ArtifactHash(artifact_id="a1", path="file", algorithm="md5", digest="a" * 32),
    ]


@pytest.mark.parametrize("value", examples(), ids=lambda value: type(value).__name__)
def test_round_trip_and_frozen(value):
    serialized = value.model_dump_json()
    assert type(value).model_validate_json(serialized) == value
    assert isinstance(json.loads(serialized), dict)
    with pytest.raises(ValidationError):
        setattr(value, next(iter(type(value).model_fields)), "changed")


def test_constants_and_stable_sigmoid():
    assert (BOGUS_LABEL, REAL_LABEL) == (0, 1)
    assert p_real_from_logit(0) == 0.5
    assert p_real_from_logit(-1000) == 0
    assert p_real_from_logit(1000) == 1
    assert prediction().p_bogus == 0.5


@pytest.mark.parametrize(
    "changes",
    [
        {"p_real": -0.1},
        {"p_real": 1.1},
        {"p_real": float("nan")},
        {"p_real": 0.2},
        {"logit": float("inf")},
        {"inference_ms": -1},
        {"inference_ms": float("nan")},
        {"tier": "huge"},
        {"score": 0.5},
        {"calibrated_p_real": 0.4},
        {"calibration_artifact_id": "cal1"},
    ],
)
def test_prediction_rejects_invalid_values(changes):
    with pytest.raises(ValidationError):
        prediction(**changes)


def test_calibrated_probability():
    result = prediction(calibrated_p_real=0.4, calibration_artifact_id="cal1")
    assert result.p_real == 0.5 and result.calibrated_p_real == 0.4


def test_candidate_lifecycle_and_identity():
    current = candidate()
    for state in (
        CandidateState.ADMITTED,
        CandidateState.RUNNING,
        CandidateState.PREDICTED,
        CandidateState.ESCALATED,
        CandidateState.QUEUED,
        CandidateState.ADMITTED,
        CandidateState.RUNNING,
        CandidateState.PREDICTED,
        CandidateState.TRIAGED,
    ):
        current = current.transition(state)
        assert (current.run_id, current.trace_id, current.candidate_id) == ("r1", "t1", "c1")
    with pytest.raises(ValueError):
        current.transition(CandidateState.QUEUED)
    with pytest.raises(ValueError):
        candidate().transition(CandidateState.TRIAGED)


@pytest.mark.parametrize(
    "contract,changes",
    [
        (4, {"selected_tier": "unknown"}),
        (5, {"outcome": "drop"}),
        (5, {"permitted_concurrency": 0}),
        (6, {"to_tier": "tiny"}),
        (6, {"to_tier": None}),
        (6, {"escalate": False}),
        (7, {"action": "delete"}),
        (7, {"threshold_artifact_sha256": "bad"}),
        (7, {"action": "discard"}),
        (7, {"output_bytes": -1}),
        (7, {"action": "compress", "output_bytes": 121}),
        (7, {"output_bytes": 119}),
        (10, {"latency_p95_ms": 0.5}),
        (3, {"cpu_percent": 101}),
        (3, {"gpu_memory_used_bytes": 20, "gpu_memory_total_bytes": 10}),
        (3, {"timestamp_utc": datetime(2026, 1, 1)}),
        (3, {"timestamp_utc": datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=5)))}),
        (9, {"status": "completed"}),
        (9, {"ended_utc": utc_now()}),
    ],
)
def test_invalid_decisions_resources_and_lifecycle(contract, changes):
    value = examples()[contract]
    with pytest.raises(ValidationError):
        type(value).model_validate({**value.model_dump(), **changes})


def test_nested_contract_mappings_are_frozen():
    manifest = examples()[9]
    with pytest.raises(TypeError):
        manifest.seeds["test"] = 2
    with pytest.raises(TypeError):
        manifest.environment["dependencies"]["test"] = "2"
    with pytest.raises(TypeError):
        manifest.environment["gpus"].append({"name": "other"})
    with pytest.raises(TypeError):
        manifest.environment["gpus"][0]["name"] = "other"


def test_tensor_objects_cannot_enter_contracts():
    event = examples()[8]
    with pytest.raises(ValidationError):
        CandidateEvent.model_validate({**event.model_dump(), "payload": {"data": object()}})
