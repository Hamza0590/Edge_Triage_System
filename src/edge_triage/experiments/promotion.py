"""Attested validation-to-test promotion. No fitting or image access in this module."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from edge_triage.config import AppConfig, load_config
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import hash_file


def deployment_identity(config: AppConfig) -> str:
    raw = json.loads(config.canonical_json())
    raw["pipeline"].pop("promotion", None)
    raw["pipeline"].pop("promotion_sha256", None)
    raw["experiment"]["name"] = "promoted_experiment"
    return hashlib.sha256(canonical(raw)).hexdigest()


def check_transition(source: AppConfig, target: AppConfig) -> None:
    if source.pipeline.partition != "validation" or target.pipeline.partition != "test":
        raise ValueError("promotion requires validation to test")
    if target.experiment.purpose != "HELD_OUT_EXPERIMENT":
        raise ValueError("promotion needs explicit held-out purpose")
    a, b = source.model_dump(mode="json"), target.model_dump(mode="json")
    for value in (a, b):
        value["pipeline"].update(partition="validation", promotion=None, promotion_sha256=None)
        value["experiment"].update(name="promotion", purpose="promotion")
        value["workload"].update(trace=None, count=1)
    if a != b:
        raise ValueError("promotion changes behavior beyond partition and declared trace")
    assert source.split.manifest and source.workload.trace and target.workload.trace
    with source.split.manifest.open(encoding="utf-8", newline="") as stream:
        splits = list(csv.DictReader(stream))
    descriptors = []
    for config in (source, target):
        assert config.workload.trace
        trace = load_trace(
            config.workload.trace,
            hash_file(source.split.manifest),
            partition=config.pipeline.partition,
        )
        expected = {
            r["candidate_id"]: int(r["index_no"])
            for r in splits
            if r["split"] == config.pipeline.partition
        }
        if (
            len(trace) != len(expected)
            or [r.source_candidate_id for r in trace] != sorted(expected)
            or any(
                r.dataset_index != expected[r.source_candidate_id]
                or r.repetition != 0
                or r.candidate_id != r.source_candidate_id
                for r in trace
            )
        ):
            raise ValueError("promotion trace must contain canonical complete unique partition")
        descriptor = json.loads(config.workload.trace.read_bytes()).get("promotion_workload")
        if not isinstance(descriptor, dict) or set(descriptor) != {"process", "interval_seconds"}:
            raise ValueError("promotion requires a declared matched workload")
        interval = descriptor["interval_seconds"]
        if (
            descriptor["process"] != "periodic"
            or not isinstance(interval, (float, int))
            or isinstance(interval, bool)
            or not math.isfinite(interval)
            or interval < 0
        ):
            raise ValueError("unsupported promotion workload")
        if any(r.relative_arrival_seconds != i * interval for i, r in enumerate(trace)):
            raise ValueError("trace differs from predeclared arrival process")
        if config.workload.count != len(trace):
            raise ValueError("trace and configured occurrence count differ")
        descriptors.append(descriptor)
    if descriptors[0] != descriptors[1]:
        raise ValueError("validation and test arrival processes differ")


def promote(source_path: Path, target: AppConfig, output: Path) -> AppConfig:
    from edge_triage.triage.artifacts import load_fate_artifact
    from edge_triage.triage.workflow import read_collection

    source = load_config(source_path)
    check_transition(source, target)
    if source.threshold.artifact is None or source.threshold.artifact_sha256 is None:
        raise ValueError("promotion requires frozen validation thresholds, including transmit-all")
    artifact = load_fate_artifact(
        source.threshold.artifact, source.threshold.artifact_sha256, source
    )
    fit_path = source_path.parent / "fit.json"
    fit = json.loads(fit_path.read_bytes())
    collection = Path(fit["collection"])
    if hash_file(collection) != fit["collection_sha256"]:
        raise ValueError("fit collection changed")
    _, evidence = read_collection(collection, source)
    if (
        fit["artifact_sha256"] != source.threshold.artifact_sha256
        or artifact.validation_records_sha256 != evidence["hashes"]["candidates.parquet"]
    ):
        raise ValueError("fit evidence does not attest threshold")
    pins = {
        str(p.resolve()): hash_file(p)
        for p in (
            source_path,
            fit_path,
            collection,
            source.workload.trace,
            target.workload.trace,
            source.threshold.artifact,
        )
        if p is not None
    }
    # Preserve the attested runtime evidence transitively, not just its index.
    pins[str(Path(json.loads(collection.read_bytes())["collection_config"]).resolve())] = hash_file(
        Path(json.loads(collection.read_bytes())["collection_config"])
    )
    for name, digest in evidence["hashes"].items():
        pins[str((Path(evidence["run"]) / name).resolve())] = digest
    value = {
        "schema_version": 1,
        "protocol": "matched_unique_partition_periodic_v1",
        "source_config": str(source_path.resolve()),
        "inputs": pins,
        "target_identity": deployment_identity(target),
        "fit_partition": "validation",
        "thresholds_frozen": True,
        "test_accessed_for_fitting": False,
        "authorization": "User requested Module11 completion, including frozen held-out evaluation",
        "limitations": "validation reuse; unknown object grouping; no recall guarantee",
    }
    write_immutable(output, canonical(value))
    raw = target.model_dump()
    raw["pipeline"].update(promotion=output.resolve(), promotion_sha256=hash_file(output))
    return AppConfig.model_validate(raw)


def verify_promotion(config: AppConfig) -> AppConfig:
    """Fail before test tensors, even for transmit-all or malformed artifacts."""
    path, digest = config.pipeline.promotion, config.pipeline.promotion_sha256
    if config.pipeline.partition != "test" or path is None or digest is None:
        raise ValueError("held-out execution requires a frozen promotion")
    if hash_file(path) != digest:
        raise ValueError("promotion checksum mismatch")
    value: dict[str, Any] = json.loads(path.read_bytes())
    if (
        value.get("protocol") != "matched_unique_partition_periodic_v1"
        or value.get("target_identity") != deployment_identity(config)
        or value.get("test_accessed_for_fitting") is not False
    ):
        raise ValueError("incompatible promotion")
    for name, expected in value["inputs"].items():
        if hash_file(Path(name)) != expected:
            raise ValueError("promotion input changed")
    source = load_config(value["source_config"])
    check_transition(source, config)
    return source
