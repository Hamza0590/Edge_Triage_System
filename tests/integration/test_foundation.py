import json

import pytest

from edge_triage.contracts import CandidateEvent, RunManifest
from edge_triage.health import dataset_hashes, doctor
from edge_triage.provenance import load_provenance
from edge_triage.runs import RunManager


@pytest.mark.dataset
def test_frozen_dataset_health_check_and_sample_run(config, isolated_config):
    paths = (config.dataset.images, config.dataset.labels)
    before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    report = doctor(config)
    assert report["ok"]
    assert [item["digest"] for item in report["dataset_hashes"]] == [
        "bbf55c1cecbdea127497d6cd346e3455",
        "cacf76a6aadd97ed18ab24b58e121bc2",
    ]
    with RunManager(isolated_config) as run:
        run.logger.bind("trace-1", "candidate-1").emit(
            module="test",
            event_type="data.reference_registered",
            reason_code="INTEGRATION",
            payload={"dataset_index": 0},
        )
    manifest = RunManifest.model_validate_json((run.directory / "manifest.json").read_text())
    assert manifest.status == "completed"
    assert len(manifest.dataset_hashes) == 2
    for line in (run.directory / "events.jsonl").read_text().splitlines():
        CandidateEvent.model_validate(json.loads(line))
    assert before == [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def test_hash_mismatch_and_missing_file_are_failures(config, tmp_path):
    wrong = tmp_path / "wrong.npy"
    wrong.write_bytes(b"not the dataset")
    raw = config.model_dump()
    raw["dataset"]["images"] = wrong
    with pytest.raises(ValueError, match="MD5 mismatch"):
        dataset_hashes(type(config).model_validate(raw))
    raw["dataset"]["images"] = tmp_path / "missing.npy"
    with pytest.raises(FileNotFoundError):
        dataset_hashes(type(config).model_validate(raw))


def test_provenance_is_explicit_about_unverified_sources(config):
    sources = load_provenance(config.paths.provenance_file)
    assert len(sources) == 12
    assert sources[0].adoption_level == "infrastructure"
    assert "engineering infrastructure" in sources[0].title
    assert {source.adoption_level for source in sources} == {
        "infrastructure",
        "adapted",
        "inspired",
    }
    assert all(source.license_status == "unknown_pending_verification" for source in sources)
