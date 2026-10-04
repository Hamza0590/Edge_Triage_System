import json

import pytest

from edge_triage.contracts import RunManifest
from edge_triage.event_logging import EventLogger
from edge_triage.runs import RunManager, append_correction


@pytest.fixture(autouse=True)
def isolate_external_probes(monkeypatch):
    # Unit tests cover lifecycle without rehashing the 386 MB dataset per test.
    # Integration tests separately exercise the real inputs and probes.
    monkeypatch.setattr("edge_triage.runs.dataset_hashes", lambda config: ())
    monkeypatch.setattr("edge_triage.runs.environment_info", lambda: {"python": "test"})
    monkeypatch.setattr("edge_triage.runs.git_state", lambda root: (None, None))


def test_creation_finalization_and_immutability(isolated_config):
    with RunManager(isolated_config) as run:
        assert run.manifest.status == "running"
        assert (run.directory / "metrics").is_dir()
        assert (run.directory / "artifacts").is_dir()
        assert not (run.directory / "candidates.parquet").exists()
        manifest_path = run.directory / "manifest.json"
    final_bytes = manifest_path.read_bytes()
    final = RunManifest.model_validate_json(final_bytes)
    assert final.status == "completed" and final.ended_utc is not None
    assert final.config_hash == isolated_config.config_hash
    assert len(final.seeds) == 6
    assert final.git_commit is None and final.git_dirty is None
    assert final.provenance_ids == ("engineering-foundation",)
    with pytest.raises(RuntimeError):
        run.finalize()
    with pytest.raises(RuntimeError):
        EventLogger(run.directory / "events.jsonl", final.run_id, final.config_hash)
    with pytest.raises(RuntimeError):
        run.logger.emit(module="test", event_type="run.extra", reason_code="TEST")
    append_correction(run.directory, author="tester", reason="annotation", correction="test only")
    assert manifest_path.read_bytes() == final_bytes
    assert json.loads((run.directory / "corrections.jsonl").read_text())["reason"] == "annotation"


def test_failure_is_recorded_and_reraised(isolated_config):
    with pytest.raises(RuntimeError, match="planned"):
        with RunManager(isolated_config) as run:
            raise RuntimeError("planned")
    assert run.manifest.status == "failed"
    records = [
        json.loads(line) for line in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    assert [item["event_type"] for item in records] == [
        "run.started",
        "error.run_failed",
        "run.failed",
    ]


@pytest.mark.parametrize("seed", [None, 17])
def test_manifest_records_stochastic_or_diagnostic_uq_seed(isolated_config, seed):
    raw = isolated_config.model_dump()
    raw["uncertainty"]["seed"] = seed
    config = type(isolated_config).model_validate(raw)
    with RunManager(config) as run:
        pass
    manifest = json.loads((run.directory / "manifest.json").read_text())
    assert "uncertainty" in manifest["seeds"] and manifest["seeds"]["uncertainty"] == seed
    assert len(manifest["seeds"]) == 6


def test_unique_run_directories_and_no_correction_on_live_run(isolated_config):
    with RunManager(isolated_config) as one, RunManager(isolated_config) as two:
        assert one.directory != two.directory
        with pytest.raises(ValueError):
            append_correction(one.directory, author="x", reason="x", correction="x")


def test_bad_dataset_does_not_create_run(isolated_config, monkeypatch):
    def fail(config):
        raise ValueError("hash mismatch")

    monkeypatch.setattr("edge_triage.runs.dataset_hashes", fail)
    with pytest.raises(ValueError):
        RunManager(isolated_config)
    assert not isolated_config.paths.runs_dir.exists()


def test_optional_artifact_hashes(isolated_config, tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"example": true}', encoding="utf-8")
    raw = isolated_config.model_dump()
    raw["split"]["manifest"] = artifact
    raw["threshold"]["artifact"] = artifact
    raw["workload"]["trace"] = artifact
    with RunManager(type(isolated_config).model_validate(raw)) as run:
        assert run.manifest.split_hash.digest == run.manifest.threshold_hashes[0].digest
        assert run.manifest.arrival_trace_hash.algorithm == "sha256"
