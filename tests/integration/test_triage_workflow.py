import json
from pathlib import Path

import pytest

from edge_triage.cli import main
from edge_triage.config import AppConfig, load_config
from edge_triage.health import hash_file
from edge_triage.triage.artifacts import load_fate_artifact
from edge_triage.triage.workflow import read_collection

ROOT = Path(__file__).resolve().parents[2]
SETTING = ROOT / "artifacts/triage/module10_v2/settings/core_tiny_sequential"


def test_saved_full_collection_is_attested():
    config = load_config(SETTING / "collection_config.yaml")
    rows, evidence = read_collection(SETTING / "collection.json", config)
    assert len(rows) == len({r.context.candidate_id for r in rows}) == 643
    assert sum(r.label for r in rows) == 329
    assert evidence["completed"] == 643 and evidence["failed"] == 0
    import pyarrow.parquet as pq

    table = pq.read_table(Path(evidence["run"]) / "candidates.parquet")
    assert table.schema.metadata[b"purpose"] == b"VALIDATION_FATE_STUDY"
    assert set(table.column("purpose").to_pylist()) == {"VALIDATION_FATE_STUDY"}


def test_collection_rejects_test_before_file_access(tmp_path):
    raw = load_config(SETTING / "collection_config.yaml").model_dump()
    raw["pipeline"]["partition"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        read_collection(tmp_path / "nonexistent.json", AppConfig.model_validate(raw))


def test_collection_rejects_other_upstream_configuration():
    raw = load_config(SETTING / "collection_config.yaml").model_dump()
    raw["calibration"]["probability_mode"] = "uncalibrated"
    with pytest.raises(ValueError, match="upstream"):
        read_collection(SETTING / "collection.json", AppConfig.model_validate(raw))


@pytest.mark.parametrize("name", ["manifest.json", "candidates.parquet", "events.jsonl"])
def test_collection_rejects_tampered_attestation(tmp_path, name):
    saved = json.loads((SETTING / "collection.json").read_bytes())
    saved["hashes"][name] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity"):
        read_collection(path, load_config(SETTING / "collection_config.yaml"))


def test_cli_fit_binds_real_collection_and_refuses_overwrite(tmp_path):
    raw = load_config(SETTING / "collection_config.yaml").model_dump()
    raw["fate"]["policy"] = "recall_constrained"
    raw["threshold"]["policy"] = "per_tier"
    config_path = tmp_path / "input.yaml"
    config_path.write_text(AppConfig.model_validate(raw).resolved_yaml(), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "search": {"min_p_real_for_transmit": [0.5], "max_uncertainty_for_discard": [0.0]},
                "bootstrap": {"repeats": 10, "seed": 20260924},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "fit"
    args = [
        "triage",
        "fit",
        "--config",
        str(config_path),
        "--collection",
        str(SETTING / "collection.json"),
        "--protocol",
        str(protocol_path),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    frozen = load_config(output / "config.yaml")
    digest = hash_file(output / "threshold.json")
    artifact = load_fate_artifact(output / "threshold.json", digest, frozen)
    assert artifact.validation_survival_recall >= 0.95
    assert (
        artifact.validation_records_sha256
        == json.loads((SETTING / "collection.json").read_bytes())["hashes"]["candidates.parquet"]
    )
    assert main(args) == 0  # Identical reruns are idempotent.
    raw["fate"]["target_survival_recall"] = 1.0
    config_path.write_text(AppConfig.model_validate(raw).resolved_yaml(), encoding="utf-8")
    assert main(args) == 1
    assert hash_file(output / "threshold.json") == digest


@pytest.mark.parametrize("field", ["hashes", "costs"])
def test_collection_rejects_incomplete_or_changed_attestation(tmp_path, field):
    saved = json.loads((SETTING / "collection.json").read_bytes())
    if field == "hashes":
        saved[field].pop("candidates.parquet")
    else:
        saved[field]["end_to_end_ms"] = 0
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="attestation mismatch"):
        read_collection(path, load_config(SETTING / "collection_config.yaml"))
