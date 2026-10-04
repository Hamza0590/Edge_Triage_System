"""Verify frozen Module 05 artifacts and calibrated real-model validation smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from edge_triage.calibration.metrics import metrics, probabilities
from edge_triage.calibration.profiling import load_profiles, summarize
from edge_triage.calibration.temperature import load_calibration
from edge_triage.calibration.workflow import code_identity
from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import canonical, frozen_split, load_raw, write_immutable
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest, ModelRegistry, read_index
from edge_triage.models.training import resolve_training_data
from edge_triage.runtime.factory import run_smoke


def verify(summary_path: Path, output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(summary_path.read_text())
    config, hashes = resolve_training_data(
        load_config(root / "configs/experiments/module05_v1.yaml")
    )
    assert summary["test_evaluated"] is False
    for path, digest in summary["artifacts"].items():
        assert hash_file(Path(path)) == digest
    index = read_index(config.models.checkpoint_index)
    registry = ModelRegistry(
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        dataset_hashes=hashes,
    )
    manifests = {}
    for tier, entry in index["checkpoints"].items():
        directory = Path(entry["directory"])
        registry.register(
            ModelTier(tier), entry["checkpoint_id"], directory, entry["manifest_sha256"]
        )
        registry.load(ModelTier(tier), entry["checkpoint_id"])
        manifests[entry["checkpoint_id"]] = (
            CheckpointManifest.model_validate_json(
                (directory / "checkpoint_manifest.json").read_bytes()
            ),
            entry["manifest_sha256"],
        )
    profiles = load_profiles(
        Path(summary["profiles"]), summary["artifacts"][summary["profiles"]], manifests
    )
    assert profiles.code_sha256 == code_identity(root)[0]
    assert profiles.family_index_sha256 == hash_file(config.models.checkpoint_index)
    assert len(profiles.profiles) == (24 if profiles.environment["cuda_supported"] else 12)
    for profile in profiles.profiles:
        raw = json.loads(Path(profile.raw_samples_path).read_text())
        for name, samples in raw["samples_ms"].items():
            assert len(samples) == profile.samples == 600
            assert summarize(samples) == profile.components[name]
        assert profile.stable and profile.stability_ratio <= config.profiling.stability_ratio
    _, labels, _ = load_raw(config)
    rows, _ = frozen_split(config, labels)
    validation = {r["candidate_id"]: r["label"] for r in rows if r["split"] == "validation"}
    excluded = {r["candidate_id"] for r in rows if r["split"] != "validation"}
    calibration_index = json.loads(Path(summary["calibration_index"]).read_text())
    quality = json.loads((Path(summary["calibration_index"]).parent / "metrics.json").read_text())
    results: dict[str, Any] = {}
    for tier in ModelTier:
        entry = calibration_index["calibrations"][tier.value]
        checkpoint = index["checkpoints"][tier.value]
        manifest, digest = manifests[checkpoint["checkpoint_id"]]
        artifact = load_calibration(Path(entry["path"]), entry["sha256"], manifest, digest)
        path = Path(entry["predictions"])
        assert hash_file(path) == entry["predictions_sha256"] == artifact.prediction_sha256
        predictions = json.loads(path.read_text())
        assert len(predictions) == len(validation) == 643
        assert {r["candidate_id"] for r in predictions} == set(validation)
        assert not ({r["candidate_id"] for r in predictions} & excluded)
        for row in predictions:
            assert row["label"] == validation[row["candidate_id"]]
            assert row["checkpoint_id"] == manifest.checkpoint_id and row["seed"] == manifest.seed
        y, z = [r["label"] for r in predictions], [r["logit"] for r in predictions]
        np.testing.assert_allclose(
            [r["p_real_uncalibrated"] for r in predictions], probabilities(z)
        )
        np.testing.assert_allclose(
            [r["p_real_calibrated"] for r in predictions], probabilities(z, artifact.temperature)
        )
        assert metrics(y, z, bins=config.calibration.bins) == artifact.before
        assert (
            metrics(y, z, temperature=artifact.temperature, bins=config.calibration.bins)
            == artifact.after
        )
        assert quality[tier.value]["uncalibrated"]["at_target_recall"]["recall"] >= 0.95
        raw = load_config(root / f"configs/experiments/smoke_{tier.value}.yaml").model_dump()
        raw["calibration"].update(
            artifact_index=Path(summary["calibration_index"]), probability_mode="calibrated"
        )
        smoke = run_smoke(AppConfig.model_validate(raw), limit=5)
        events = [json.loads(line) for line in (smoke / "events.jsonl").read_text().splitlines()]
        outputs = [e for e in events if e["event_type"] == "inference.predicted"]
        assert len(outputs) == 5
        for event in outputs:
            prediction = event["payload"]
            assert prediction["calibration_artifact_id"] == artifact.artifact_id
            assert (
                abs(
                    prediction["calibrated_p_real"]
                    - float(probabilities(prediction["logit"], artifact.temperature))
                )
                < 1e-12
            )
        results[tier.value] = {
            "temperature": artifact.temperature,
            "mode": artifact.recommended_mode,
            "predictions": len(predictions),
            "smoke": str(smoke),
            "smoke_manifest_sha256": hash_file(smoke / "manifest.json"),
        }
    write_immutable(
        output,
        canonical(
            {
                "schema_version": 1,
                "verified": True,
                "test_evaluated": False,
                "summary_sha256": hash_file(summary_path),
                "profiles": len(profiles.profiles),
                "gate_b": summary["gate_b"]["pass"],
                "tiers": results,
                "dataset_hashes": hashes,
                "code_sha256": profiles.code_sha256,
            }
        ),
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.summary, args.output)
