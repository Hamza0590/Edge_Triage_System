"""Attested complete-validation collection, frozen fitting and runtime verification."""

from __future__ import annotations

import contextlib
import csv
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import PredictionResult, UncertaintyResult
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.workloads import TraceRecord
from edge_triage.health import hash_file
from edge_triage.models.registry import read_index
from edge_triage.models.training import resolve_training_data
from edge_triage.runtime.interfaces import FateContext
from edge_triage.runtime.records import CandidateRecord
from edge_triage.runtime.streaming import run_concurrent
from edge_triage.triage.artifacts import (
    FateOperatingPoint,
    FateThresholdArtifact,
    PerTierThresholdPolicy,
    TierFateDefinition,
    load_fate_artifact,
    system_identity,
)
from edge_triage.triage.fitting import (
    ValidationFateRow,
    _validation_rows,
    bootstrap_fates,
    evaluate_fates,
    fit_thresholds,
)
from edge_triage.triage.policies import (
    LogicalByteAccounting,
    PerTierRecallConstrainedPolicy,
    TransmitAllPolicy,
)


def save_json(path: Path, value: Any) -> None:
    write_immutable(path, canonical(value))


def upstream_identity(config: AppConfig) -> str:
    raw = config.model_dump()
    raw["fate"].update(policy="transmit_all")
    raw["threshold"].update(policy="common", artifact=None, artifact_sha256=None)
    return system_identity(AppConfig.model_validate(raw))


def validation_trace(config: AppConfig, directory: Path) -> Path:
    if config.pipeline.partition != "validation":
        raise ValueError("fate collection is validation-only")
    config, _ = resolve_training_data(config)
    assert config.split.manifest is not None
    with config.split.manifest.open(encoding="utf-8", newline="") as stream:
        rows = sorted(
            (r for r in csv.DictReader(stream) if r["split"] == "validation"),
            key=lambda r: r["candidate_id"],
        )
    trace = {
        "schema_version": 1,
        "partition": "validation",
        "split_sha256": hash_file(config.split.manifest),
        "purpose": "COMPLETE_UNIQUE_VALIDATION_FOR_FATE_FITTING",
        "statistical_independence": "unique supplied rows; repeated-object independence unknown",
        "records": [
            TraceRecord(
                trace_position=i,
                candidate_id=row["candidate_id"],
                source_candidate_id=row["candidate_id"],
                dataset_index=int(row["index_no"]),
                relative_arrival_seconds=0,
                workload_phase="backlog",
                seed=config.workload.seed,
                repetition=0,
            ).model_dump(mode="json")
            for i, row in enumerate(rows)
        ],
    }
    path = directory / "validation_trace.json"
    save_json(path, trace)
    write_immutable(path.with_suffix(".sha256"), (hash_file(path) + "\n").encode())
    return path.resolve()


def row_evidence(row: CandidateRecord) -> ValidationFateRow:
    predictions = json.loads(row.prediction_chain_json)
    uncertainty = json.loads(row.uncertainty_chain_json)
    prediction = PredictionResult.model_validate(predictions[-1]) if predictions else None
    uq = UncertaintyResult.model_validate(uncertainty[-1]) if uncertainty else None
    if prediction is not None and prediction.candidate_id != row.candidate_id:
        raise ValueError("final prediction identity mismatch")
    return ValidationFateRow(
        context=FateContext(
            candidate_id=row.candidate_id,
            prediction=prediction,
            uncertainty=uq,
            conservative_fallback=bool(row.conservative_fallback),
            failed=row.status == "failed",
        ),
        label=row.label,
        original_bytes=120000,
        end_to_end_ms=row.end_to_end_ms,
    )


def inspect_run(config: AppConfig, run: Path) -> tuple[list[CandidateRecord], dict[str, Any]]:
    if config.pipeline.partition != "validation":
        raise ValueError("fate study cannot inspect non-validation runs")
    manifest = json.loads((run / "manifest.json").read_bytes())
    if manifest["status"] != "completed" or manifest["config_hash"] != config.config_hash:
        raise ValueError("run is incomplete or config identity mismatches")
    assert config.split.manifest is not None and config.workload.trace is not None
    summary = json.loads((run / "metrics/concurrent_summary.json").read_bytes())
    if summary["trace_sha256"] != hash_file(config.workload.trace):
        raise ValueError("runtime trace identity mismatch")
    records = [
        CandidateRecord.model_validate(r)
        for r in pq.read_table(run / "candidates.parquet").to_pylist()
    ]
    _validation_rows(
        (row_evidence(r) for r in records),
        "validation",
        config.split.manifest,
        hash_file(config.split.manifest),
    )
    if summary["completed"] + summary["failed"] != len(records):
        raise ValueError("candidate losses in runtime collection")
    if any(summary["queue"][k] for k in ("dropped", "rejected", "cancelled")):
        raise ValueError("lossy validation collection")
    if summary["availability"]["active_jobs_total"] != 0:
        raise ValueError("runtime failed to drain")
    final_ids = []
    resource_ids = set()
    for line in (run / "resources.jsonl").read_text().splitlines():
        resource_ids.add(json.loads(line)["snapshot_id"])
    for line in (run / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event_type"] == "inference.candidate_final":
            final_ids.append(event["candidate_id"])
    if sorted(final_ids) != sorted(r.candidate_id for r in records):
        raise ValueError("candidate final events mismatch")
    for row in records:
        if row.input_bytes != 120000 + config.fate.metadata_bytes:
            raise ValueError("unexpected logical raw payload")
        for snapshot in json.loads(row.resource_snapshot_json):
            if snapshot.get("snapshot_id") and snapshot["snapshot_id"] not in resource_ids:
                raise ValueError("missing persisted resource snapshot")
    names = (
        "manifest.json",
        "config.resolved.yaml",
        "events.jsonl",
        "resources.jsonl",
        "candidates.parquet",
        "metrics/concurrent_summary.json",
    )
    evidence = {
        "run": str(run.resolve()),
        "config_sha256": config.config_hash,
        "upstream_sha256": upstream_identity(config),
        "hashes": {name: hash_file(run / name) for name in names},
        "completed": summary["completed"],
        "failed": summary["failed"],
        "candidates": len(records),
        "summary": summary,
        "test_evaluated": False,
        "costs": {
            name: sum(getattr(r, name) for r in records)
            for name in (
                "inference_ms",
                "uq_ms",
                "admission_ms",
                "executor_ms",
                "escalation_decision_ms",
                "fate_ms",
                "end_to_end_ms",
                "additional_forward_passes",
            )
        },
    }
    return records, evidence


def collect(config: AppConfig, directory: Path, *, warmup_forwards_per_tier: int = 0) -> Path:
    if config.pipeline.partition != "validation" or config.fate.policy != "transmit_all":
        raise ValueError("collection requires validation and explicit transmit-all fate")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "collection.json"
    if path.exists():
        _, evidence = read_collection(path, config)
        if (
            evidence["summary"].get("warmup", {}).get("forwards_per_tier", 0)
            != warmup_forwards_per_tier
        ):
            raise ValueError("collection warm-up protocol differs")
        return path
    write_immutable(directory / "collection_config.yaml", config.resolved_yaml().encode())
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        run = run_concurrent(config, warmup_forwards_per_tier=warmup_forwards_per_tier)
    _, evidence = inspect_run(config, run)
    evidence["collection_config"] = str((directory / "collection_config.yaml").resolve())
    save_json(path, evidence)
    return path


def read_collection(
    path: Path, target: AppConfig
) -> tuple[list[ValidationFateRow], dict[str, Any]]:
    # Guard before reading candidate evidence, even if the supplied attestation is invalid.
    if target.pipeline.partition != "validation":
        raise ValueError("fitting collections are validation-only")
    evidence = json.loads(path.read_bytes())
    config = load_config(evidence["collection_config"])
    if config.fate.policy != "transmit_all" or upstream_identity(target) != upstream_identity(
        config
    ):
        raise ValueError("collection does not match complete upstream configuration")
    run = Path(evidence["run"])
    for name, digest in evidence["hashes"].items():
        if hash_file(run / name) != digest:
            raise ValueError(f"collection integrity mismatch: {name}")
    records, actual = inspect_run(config, run)
    if actual != {key: value for key, value in evidence.items() if key != "collection_config"}:
        raise ValueError("collection attestation mismatch")
    return [row_evidence(r) for r in records], evidence


def tier_definitions(config: AppConfig) -> tuple[TierFateDefinition, ...]:
    assert config.models.checkpoint_index and config.calibration.artifact_index
    family = read_index(config.models.checkpoint_index)
    calibration = json.loads(config.calibration.artifact_index.read_bytes())
    result = []
    for tier in config.models.tier_portfolio:
        entry = calibration["calibrations"][tier.value]
        if hash_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("calibration hash mismatch")
        cal = json.loads(Path(entry["path"]).read_bytes())
        mode = (
            cal["recommended_mode"]
            if config.calibration.probability_mode == "recommended"
            else config.calibration.probability_mode
        )
        result.append(
            TierFateDefinition(
                tier=tier,
                checkpoint_id=family["checkpoints"][tier.value]["checkpoint_id"],
                probability_mode=mode,
                calibration_artifact_id=cal["artifact_id"] if mode == "calibrated" else None,
                operating_point=FateOperatingPoint(),
            )
        )
    return tuple(result)


def fit_collection(
    collection: Path, target: AppConfig, output: Path, protocol: dict[str, Any]
) -> Path:
    rows, evidence = read_collection(collection, target)
    assert target.split.manifest and target.models.checkpoint_index
    assert target.preprocessing.artifact and target.calibration.artifact_index
    accounting = LogicalByteAccounting(
        compression_fraction=target.fate.compression_fraction,
        metadata_bytes=target.fate.metadata_bytes,
    )
    tiers = tier_definitions(target)
    is_all = target.fate.policy == "transmit_all"
    binary = target.fate.policy == "binary"
    if is_all:
        tiers = tuple(
            TierFateDefinition.model_validate({**s.model_dump(), "discard_supported": False})
            for s in tiers
        )
        metrics = evaluate_fates(rows, tiers, accounting)
        objective, searched = "Transmit-all bandwidth upper baseline", 0
    else:
        result = fit_thresholds(
            rows,
            partition="validation",
            split_path=target.split.manifest,
            split_sha256=hash_file(target.split.manifest),
            tiers=tiers,
            accounting=accounting,
            target_survival_recall=target.fate.target_survival_recall,
            common=target.threshold.policy == "common",
            monotonic=True,
            binary=binary,
            transmit_grid=tuple(protocol["search"]["min_p_real_for_transmit"]),
            uncertainty_grid=tuple(protocol["search"]["max_uncertainty_for_discard"]),
        )
        tiers, metrics, objective, searched = (
            result.tiers,
            result.metrics,
            result.objective,
            result.searched_options,
        )
    artifact = FateThresholdArtifact(
        artifact_id=f"module10:{system_identity(target)[:20]}",
        threshold_policy=target.threshold.policy,
        system_sha256=system_identity(target),
        family_sha256=hash_file(target.models.checkpoint_index),
        split_sha256=hash_file(target.split.manifest),
        preprocessing_sha256=hash_file(target.preprocessing.artifact),
        calibration_index_sha256=hash_file(target.calibration.artifact_index),
        validation_records_sha256=evidence["hashes"]["candidates.parquet"],
        target_survival_recall=target.fate.target_survival_recall,
        validation_survival_recall=metrics["survival_recall"],
        objective=objective,
        validation_metrics=metrics,
        tiers=tiers,
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "threshold.json"
    PerTierThresholdPolicy(artifact).persist(path)
    raw = target.model_dump()
    raw["threshold"].update(artifact=path.resolve(), artifact_sha256=hash_file(path))
    frozen = AppConfig.model_validate(raw)
    write_immutable(output / "config.yaml", frozen.resolved_yaml().encode())
    load_fate_artifact(path, hash_file(path), frozen)
    report = {
        "status": "validation_feasible",
        "collection": str(collection.resolve()),
        "collection_sha256": hash_file(collection),
        "system_sha256": artifact.system_sha256,
        "artifact_sha256": hash_file(path),
        "metrics": metrics,
        "searched_options": searched,
        "bootstrap": bootstrap_fates(
            rows,
            tiers,
            accounting,
            repeats=protocol["bootstrap"]["repeats"],
            seed=protocol["bootstrap"]["seed"],
            binary=binary,
        ),
        "upstream_costs": evidence["costs"],
        "test_evaluated": False,
    }
    save_json(output / "fit.json", report)
    return output / "config.yaml"


def verify_frozen_run(config: AppConfig, directory: Path) -> dict[str, Any]:
    assert config.threshold.artifact and config.threshold.artifact_sha256
    artifact = load_fate_artifact(
        config.threshold.artifact, config.threshold.artifact_sha256, config
    )
    records, evidence = inspect_run(config, directory)
    thresholds = PerTierThresholdPolicy(artifact)
    accounting = LogicalByteAccounting(
        compression_fraction=config.fate.compression_fraction,
        metadata_bytes=config.fate.metadata_bytes,
    )
    policy = (
        TransmitAllPolicy(accounting)
        if config.fate.policy == "transmit_all"
        else PerTierRecallConstrainedPolicy(accounting, config.fate.policy == "binary")
    )
    for record in records:
        row = row_evidence(record)
        prediction = row.context.prediction
        expected = policy.decide_context(
            row.context,
            thresholds.resolve(prediction.tier) if prediction and not row.context.failed else None,
            row.original_bytes,
        )
        if (record.fate, record.output_bytes) != (expected.action, expected.output_bytes):
            raise ValueError("runtime fate disagrees with frozen policy")
        if (
            not row.context.failed
            and record.threshold_artifact_sha256 != config.threshold.artifact_sha256
        ):
            raise ValueError("runtime threshold hash mismatch")
    metrics = evaluate_fates(
        [row_evidence(r) for r in records],
        artifact.tiers,
        accounting,
        binary=config.fate.policy == "binary",
    )
    evidence["metrics"] = metrics
    evidence["status"] = (
        "verified_feasible"
        if metrics["survival_recall"] >= config.fate.target_survival_recall
        else "infeasible_on_runtime_replay_do_not_deploy"
    )
    evidence["artifact_sha256"] = hash_file(config.threshold.artifact)
    return evidence


def replay(config: AppConfig, output: Path) -> Path:
    if output.exists():
        saved = json.loads(output.read_bytes())
        actual = verify_frozen_run(config, Path(saved["run"]))
        if actual != saved:
            raise ValueError("saved replay changed")
        return output
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        run = run_concurrent(config)
    save_json(output, verify_frozen_run(config, run))
    return output
