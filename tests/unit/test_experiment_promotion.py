import json

import pytest

from edge_triage.config import AppConfig, load_config
from edge_triage.experiments.matrix import enforce_comparison
from edge_triage.experiments.promotion import check_transition, verify_promotion
from edge_triage.runtime.concurrent import FixedMemoryAdmission, create_executor
from edge_triage.runtime.streaming import run_concurrent
from edge_triage.scheduling.policies import (
    MemoryContentionAwareAdmissionPolicy,
    ResourceQueueAwareTierPolicy,
)
from edge_triage.triage.artifacts import load_fate_artifact
from tests.fixtures.experiments import ROOT
from tests.unit.test_scheduling import candidate, context, snapshot


def test_legacy_fate_config_identity_is_preserved():
    config = load_config(ROOT / "configs/experiments/module10_proposed_v2.yaml")
    load_fate_artifact(config.threshold.artifact, config.threshold.artifact_sha256, config)
    raw = json.loads(config.canonical_json())
    assert "concurrent_policy" not in raw["admission"] and "promotion" not in raw["pipeline"]


def test_queue_axis_removes_queue_signal_but_keeps_memory_guard():
    policy = ResourceQueueAwareTierPolicy(context(), queue_aware=False)
    assert policy.select(candidate(), snapshot(queue_length=100)).selected_tier.value == "large"
    assert (
        policy.select(candidate(), snapshot(queue_length=100, available_ram_bytes=1)).selected_tier
        is None
    )
    aware = ResourceQueueAwareTierPolicy(context())
    assert aware.select(candidate(), snapshot(queue_length=100)).selected_tier.value == "tiny"


@pytest.mark.parametrize("mode", ["sequential", "fixed_concurrency", "dynamic_concurrency"])
@pytest.mark.parametrize("admission", ["fixed_memory", "memory_contention_aware"])
def test_admission_is_independent_of_execution(mode, admission):
    config = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml")
    raw = config.model_dump()
    raw["admission"].update(execution_mode=mode, concurrent_policy=admission)
    changed = AppConfig.model_validate(raw)
    engine = create_executor(changed, snapshot)
    expected = (
        FixedMemoryAdmission
        if admission == "fixed_memory"
        else MemoryContentionAwareAdmissionPolicy
    )
    assert isinstance(engine.admission, expected)
    other = changed.model_dump()
    other["admission"]["concurrent_policy"] = "fixed_memory"
    enforce_comparison([changed, AppConfig.model_validate(other)], ("admission_policy",))


def test_test_guard_precedes_data_or_runtime_access(monkeypatch):
    config = load_config(ROOT / "configs/experiments/module10_proposed_v2.yaml")
    raw = config.model_dump()
    raw["pipeline"].update(partition="test")
    config = AppConfig.model_validate(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("test data accessed before promotion")

    monkeypatch.setattr("edge_triage.runtime.streaming._run_stream", forbidden)
    with pytest.raises(ValueError, match="frozen promotion"):
        run_concurrent(config)
    with pytest.raises(ValueError, match="frozen promotion"):
        load_fate_artifact(config.threshold.artifact, config.threshold.artifact_sha256, config)
    with pytest.raises(ValueError, match="frozen promotion"):
        verify_promotion(config)


def test_promotion_rejects_behavior_change_before_trace_access():
    config = load_config(ROOT / "configs/experiments/module10_proposed_v2.yaml")
    raw = config.model_dump()
    raw["pipeline"]["partition"] = "test"
    raw["experiment"]["purpose"] = "HELD_OUT_EXPERIMENT"
    raw["fate"]["compression_fraction"] = 0.5
    with pytest.raises(ValueError, match="changes behavior"):
        check_transition(config, AppConfig.model_validate(raw))


def test_promotion_roundtrip_and_tamper_preaccess(tmp_path, monkeypatch):
    import csv

    from edge_triage.data.artifacts import canonical
    from edge_triage.experiments.promotion import promote
    from edge_triage.health import hash_file
    from edge_triage.triage.artifacts import system_identity

    original = load_config(ROOT / "configs/experiments/module10_proposed_v2.yaml")
    with original.split.manifest.open(newline="", encoding="utf-8") as stream:
        split = list(csv.DictReader(stream))
    traces = {}
    for partition in ("validation", "test"):
        rows = sorted(
            (r for r in split if r["split"] == partition), key=lambda r: r["candidate_id"]
        )
        records = [
            dict(
                trace_position=i,
                candidate_id=r["candidate_id"],
                source_candidate_id=r["candidate_id"],
                dataset_index=int(r["index_no"]),
                relative_arrival_seconds=0,
                workload_phase="backlog",
                seed=20260924,
                repetition=0,
            )
            for i, r in enumerate(rows)
        ]
        path = tmp_path / f"{partition}.json"
        path.write_bytes(
            canonical(
                dict(
                    schema_version=1,
                    partition=partition,
                    split_sha256=hash_file(original.split.manifest),
                    records=records,
                    promotion_workload={"process": "periodic", "interval_seconds": 0},
                )
            )
        )
        path.with_suffix(".sha256").write_text(hash_file(path))
        traces[partition] = (path, len(rows))
    raw = original.model_dump()
    raw["workload"].update(trace=traces["validation"][0], count=traces["validation"][1])
    artifact = json.loads(original.threshold.artifact.read_bytes())
    candidate_evidence = tmp_path / "candidates.parquet"
    candidate_evidence.write_bytes(b"TEST_ONLY_SYNTHETIC_ATTESTATION")
    artifact.update(
        system_sha256=system_identity(AppConfig.model_validate(raw)),
        artifact_id="TEST_ONLY_PROMOTION",
        validation_records_sha256=hash_file(candidate_evidence),
    )
    artifact_path = tmp_path / "threshold.json"
    artifact_path.write_bytes(canonical(artifact))
    raw["threshold"].update(artifact=artifact_path, artifact_sha256=hash_file(artifact_path))
    source = AppConfig.model_validate(raw)
    source_path = tmp_path / "config.yaml"
    source_path.write_text(source.resolved_yaml())
    collection = tmp_path / "collection.json"
    collection_config = tmp_path / "collection_config.yaml"
    collection_config.write_text(source.resolved_yaml())
    collection.write_text(json.dumps({"collection_config": str(collection_config)}))
    (tmp_path / "fit.json").write_bytes(
        canonical(
            dict(
                collection=str(collection),
                collection_sha256=hash_file(collection),
                artifact_sha256=hash_file(artifact_path),
            )
        )
    )
    # Only this synthetic attestation is mocked; all transition/file/config guards are real.
    monkeypatch.setattr(
        "edge_triage.triage.workflow.read_collection",
        lambda *args: (
            [],
            {
                "run": str(tmp_path),
                "hashes": {"candidates.parquet": artifact["validation_records_sha256"]},
            },
        ),
    )
    raw["pipeline"]["partition"] = "test"
    raw["experiment"]["purpose"] = "HELD_OUT_EXPERIMENT"
    raw["workload"].update(trace=traces["test"][0], count=traces["test"][1])
    target = promote(source_path, AppConfig.model_validate(raw), tmp_path / "promotion.json")
    assert verify_promotion(target) == source
    assert (
        load_fate_artifact(artifact_path, hash_file(artifact_path), target).artifact_id
        == "TEST_ONLY_PROMOTION"
    )
    changed = target.model_dump()
    changed["admission"]["fixed_concurrency"] += 1
    with pytest.raises(ValueError, match="incompatible promotion"):
        verify_promotion(AppConfig.model_validate(changed))
    candidate_evidence.write_bytes(b"TAMPERED")
    with pytest.raises(ValueError, match="input changed"):
        verify_promotion(target)
    candidate_evidence.write_bytes(b"TEST_ONLY_SYNTHETIC_ATTESTATION")
    traces["test"][0].write_bytes(traces["test"][0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="input changed"):
        verify_promotion(target)
