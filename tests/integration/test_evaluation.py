import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from edge_triage.calibration.profiling import (
    MeasuredProfile,
    ProfileBundle,
    load_profiles,
    summarize,
)
from edge_triage.calibration.workflow import evaluate_saved, gate_b, validation_predictions
from edge_triage.config import load_config
from edge_triage.contracts import ModelTier, utc_now
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest, ModelRegistry, read_index
from edge_triage.models.training import resolve_training_data

ROOT = Path(__file__).resolve().parents[2]


def manifest_and_registry():
    config, hashes = resolve_training_data(
        load_config(ROOT / "configs/experiments/module05_v1.yaml")
    )
    index = read_index(config.models.checkpoint_index)
    entry = index["checkpoints"]["tiny"]
    directory = Path(entry["directory"])
    manifest = CheckpointManifest.model_validate_json(
        (directory / "checkpoint_manifest.json").read_bytes()
    )
    registry = ModelRegistry(
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        dataset_hashes=hashes,
    )
    registry.register(ModelTier.TINY, entry["checkpoint_id"], directory, entry["manifest_sha256"])
    return config, manifest, registry, entry


def test_validation_collection_saved_evaluator_and_loader_guard(tmp_path, monkeypatch):
    config, manifest, registry, entry = manifest_and_registry()
    from edge_triage.calibration import workflow

    original = workflow.make_loader
    accesses = []

    def guarded_loader(config, partition, **kwargs):
        assert partition == "validation"
        accesses.append(partition)
        return original(config, partition, **kwargs)

    monkeypatch.setattr(workflow, "make_loader", guarded_loader)
    torch.set_num_threads(1)
    rows = validation_predictions(
        config, registry, ModelTier.TINY, entry["checkpoint_id"], "eval-test"
    )
    assert accesses == ["validation"]
    assert len(rows) == len({r["candidate_id"] for r in rows}) == 643
    expected = json.loads((Path(entry["directory"]) / "metrics.json").read_text())["validation"]
    path = tmp_path / "predictions.json"
    write_immutable(path, canonical(rows))
    config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"bootstrap_repeats": 20})}
    )
    result = evaluate_saved(path, config, 1)
    assert result["uncalibrated"]["at_0_5"]["average_precision"] == pytest.approx(
        expected["average_precision"]
    )
    assert result["uncalibrated"]["at_target_recall"]["recall"] >= 0.95
    assert result["uncalibrated"] == result["calibrated"]


def test_profile_schema_uniqueness_integrity_and_checkpoint_compatibility(tmp_path):
    config, manifest, _, entry = manifest_and_registry()
    raw = tmp_path / "samples.json"
    write_immutable(raw, b"[1,2,3]")
    latency = summarize([1, 2, 3])
    p = MeasuredProfile(
        profile_id="p",
        tier=ModelTier.TINY,
        checkpoint_id=manifest.checkpoint_id,
        device="cpu",
        batch_size=1,
        concurrency=1,
        input_shape=(3, 30, 30),
        latency_p50_ms=2,
        latency_p95_ms=2.9,
        peak_memory_bytes=100,
        samples=3,
        measured_utc=utc_now(),
        hardware_id="h",
        checkpoint_manifest_sha256=entry["manifest_sha256"],
        split_sha256=manifest.split_sha256,
        preprocessing_sha256=manifest.preprocessing_sha256,
        parameter_count=manifest.parameter_count,
        checkpoint_bytes=10,
        load_ms=2,
        warmup=2,
        rounds=2,
        components={"end_to_end": latency},
        throughput_candidates_s=500,
        amortized_ms_per_candidate=2,
        batch_completion_median_ms=2,
        cuda_allocated_peak_bytes=None,
        cuda_reserved_peak_bytes=None,
        host_rss_before_bytes=90,
        host_rss_after_bytes=100,
        host_rss_sampled_peak_bytes=100,
        round_medians=(2, 2),
        stability_ratio=1,
        stable=True,
        raw_samples_path=str(raw),
        raw_samples_sha256=hash_file(raw),
    )
    bundle = ProfileBundle(
        config_sha256="c",
        code_sha256="c",
        family_index_sha256="i",
        profiles=(p,),
        environment={},
        methodology={},
        overhead={},
    )
    with pytest.raises(ValidationError, match="duplicate"):
        ProfileBundle.model_validate({**bundle.model_dump(), "profiles": [p, p]})
    path = tmp_path / "profile.json"
    write_immutable(path, canonical(bundle.model_dump(mode="json")))
    manifests = {manifest.checkpoint_id: (manifest, entry["manifest_sha256"])}
    assert load_profiles(path, hash_file(path), manifests) == bundle
    with pytest.raises(ValueError, match="incompatible"):
        load_profiles(path, hash_file(path), {manifest.checkpoint_id: (manifest, "bad")})
    raw.write_bytes(b"[0]")
    with pytest.raises(ValueError, match="raw timing"):
        load_profiles(path, hash_file(path), manifests)
    quality = {
        tier: {
            "uncalibrated": {
                "at_target_recall": {"average_precision": ap, "precision": 1.0, "recall": 0.96}
            }
        }
        for tier, ap in [("tiny", 0.98), ("large", 0.99)]
    }
    large = p.model_copy(update={"tier": ModelTier.LARGE, "latency_p50_ms": 4.0})
    assert gate_b(quality, [p, large], config)["pass"]
    assert not gate_b(quality, [p, large.model_copy(update={"stable": False})], config)["pass"]
