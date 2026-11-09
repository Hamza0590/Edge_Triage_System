"""Freeze, execute and verify independent common-pipeline experiment subruns."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from edge_triage.config import AppConfig, load_config
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.components import COMPONENT_FILES, run_component
from edge_triage.experiments.matrix import ExperimentMatrix, load_matrix, validate_matrix
from edge_triage.experiments.reporting import distribution, summarize_records
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import hash_file
from edge_triage.runtime.records import CandidateRecord
from edge_triage.runtime.streaming import run_concurrent
from edge_triage.triage.artifacts import PerTierThresholdPolicy, load_fate_artifact
from edge_triage.triage.policies import (
    LogicalByteAccounting,
    PerTierRecallConstrainedPolicy,
    TransmitAllPolicy,
)
from edge_triage.triage.workflow import row_evidence

RUN_FILES = (
    "manifest.json",
    "config.resolved.yaml",
    "candidates.parquet",
    "events.jsonl",
    "resources.jsonl",
    "metrics/concurrent_summary.json",
)


def source_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hash_file(path)
        for path in sorted((root / "src/edge_triage").rglob("*.py"))
    }


def freeze_matrix(path: Path) -> tuple[ExperimentMatrix, dict[str, Any]]:
    matrix = load_matrix(path)
    resolved = validate_matrix(matrix)
    output = matrix.output_dir
    # A result directory may not contain project source, data or any pinned input.
    protected = [matrix.project_root / "src", matrix.project_root / "tests", path.resolve()]
    protected.extend(Path(p) for p in resolved["inputs"])
    if any(p == output or output in p.parents for p in protected):
        raise ValueError("experiment output would contain protected inputs")
    resolved["source_hashes"] = source_inventory(matrix.project_root)
    resolved["software"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pyarrow", "pydantic", "PyYAML", "psutil")
        },
    }
    resolved["runtime_scope"] = "frozen_partition_harness_v2"
    resolved["test_authorization"] = (
        "frozen matrix and separately fitted validation thresholds; no tuning after test"
        if matrix.partition == "test"
        else None
    )
    # Write this first: a changed matrix/source cannot add files to an old study.
    write_immutable(output / "matrix.resolved.json", canonical(resolved))
    write_immutable(output / "matrix.yaml", path.read_bytes())
    for case in matrix.cases:
        if case.config and case.status == "ready":
            config = load_config(case.config.path)
            write_immutable(output / "configs" / f"{case.id}.yaml", config.resolved_yaml().encode())
    return matrix, resolved


def verify_hashes(base: Path, inventory: dict[str, str]) -> None:
    for name, digest in inventory.items():
        if hash_file(base / name) != digest:
            raise ValueError(f"saved experiment integrity mismatch: {name}")


def read_completed(directory: Path, study_sha256: str, subrun_id: str) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((directory / "complete.json").read_bytes())
    if value["study_sha256"] != study_sha256 or value["subrun_id"] != subrun_id:
        raise ValueError("completed subrun belongs to an incompatible matrix")
    expected_files = COMPONENT_FILES if value.get("kind") == "component" else RUN_FILES
    if value.get("schema_version") != 1 or set(value["run_hashes"]) != set(expected_files):
        raise ValueError("incomplete completion inventory")
    metrics_path = (directory / value["metrics_file"]).resolve()
    if not metrics_path.is_relative_to(directory.resolve()):
        raise ValueError("completion metrics path escapes subrun")
    verify_hashes(Path(value["run"]), value["run_hashes"])
    verify_hashes(directory, {value["metrics_file"]: value["metrics_sha256"]})
    return value


def inspect_run(config: AppConfig, run: Path, matrix: ExperimentMatrix) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_bytes())
    if manifest["status"] != "completed" or manifest["config_hash"] != config.config_hash:
        raise ValueError("runtime incomplete or configuration mismatch")
    if load_config(run / "config.resolved.yaml").config_hash != config.config_hash:
        raise ValueError("persisted runtime configuration mismatch")
    assert config.workload.trace and config.split.manifest
    trace = load_trace(
        config.workload.trace, hash_file(config.split.manifest), partition=matrix.partition
    )
    runtime = json.loads((run / "metrics/concurrent_summary.json").read_bytes())
    if runtime["trace_sha256"] != hash_file(config.workload.trace):
        raise ValueError("runtime trace mismatch")
    if runtime["warmup"]["forwards_per_tier"] != matrix.warmup_forwards_per_tier:
        raise ValueError("warm-up protocol mismatch")
    if runtime["availability"]["active_jobs_total"] != 0:
        raise ValueError("executor did not drain")
    if any(runtime["queue"][key] for key in ("rejected", "dropped", "cancelled")):
        raise ValueError("candidate loss in experiment")
    rows = [
        CandidateRecord.model_validate(row)
        for row in pq.read_table(run / "candidates.parquet").to_pylist()
    ]
    if len(rows) != len(trace) or runtime["completed"] + runtime["failed"] != len(rows):
        raise ValueError("runtime candidate count mismatch")
    if sum(r.status == "failed" for r in rows) != runtime["failed"]:
        raise ValueError("runtime failure count mismatch")
    with config.split.manifest.open(encoding="utf-8", newline="") as stream:
        labels = {
            r["candidate_id"]: int(r["label"])
            for r in csv.DictReader(stream)
            if r["split"] == matrix.partition
        }
    sources = {r.candidate_id: r.source_candidate_id for r in trace}
    final_ids = []
    for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] == "inference.candidate_final":
            final_ids.append(event["candidate_id"])
    if sorted(final_ids) != sorted(sources):
        raise ValueError("final events missing or duplicated")
    resources = [
        json.loads(line)
        for line in (run / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    resource_ids = {r["snapshot_id"] for r in resources}
    thresholds = None
    if config.threshold.artifact and config.threshold.artifact_sha256:
        thresholds = PerTierThresholdPolicy(
            load_fate_artifact(config.threshold.artifact, config.threshold.artifact_sha256, config)
        )
    accounting = LogicalByteAccounting(
        compression_fraction=config.fate.compression_fraction,
        metadata_bytes=config.fate.metadata_bytes,
    )
    fate_policy = (
        TransmitAllPolicy(accounting)
        if config.fate.policy == "transmit_all"
        else PerTierRecallConstrainedPolicy(accounting, config.fate.policy == "binary")
    )
    for row in rows:
        if row.candidate_id not in sources or row.label != labels[sources[row.candidate_id]]:
            raise ValueError("candidate source/label identity mismatch")
        if row.run_id != manifest["run_id"]:
            raise ValueError("candidate run identity mismatch")
        if row.purpose != config.experiment.purpose:
            raise ValueError("candidate purpose mismatch")
        if (
            row.threshold_artifact_sha256 != config.threshold.artifact_sha256
            and config.threshold.artifact
            and row.status != "failed"
        ):
            raise ValueError("candidate did not use frozen fate artifact")
        if thresholds is not None:
            evidence = row_evidence(row)
            prediction = evidence.context.prediction
            expected = fate_policy.decide_context(
                evidence.context,
                thresholds.resolve(prediction.tier)
                if prediction and not evidence.context.failed
                else None,
                evidence.original_bytes,
            )
            if (row.fate, row.input_bytes, row.output_bytes) != (
                expected.action,
                expected.input_bytes,
                expected.output_bytes,
            ):
                raise ValueError("runtime fate/bytes differ from frozen policy")
        for snapshot in json.loads(row.resource_snapshot_json):
            if snapshot.get("snapshot_id") and snapshot["snapshot_id"] not in resource_ids:
                raise ValueError("missing resource evidence")
    result = summarize_records(
        rows,
        sources,
        runtime,
        resources,
        unique_validation=matrix.purpose in {"unique_validation", "unique_test"},
        partition=matrix.partition,
        target=matrix.target_survival_recall,
        bootstrap_repeats=matrix.bootstrap_repeats,
        seed=matrix.bootstrap_seed,
    )
    from edge_triage.scheduling.profiles import scheduler_profiles

    profiles = scheduler_profiles(config)
    jobs = [
        job
        for row in rows
        for job in json.loads(row.execution_chain_json)
        if job["started_ns"] is not None
    ]
    result["inference_degradation"] = {
        tier.value: distribution(
            [job["inference_ms"] / profile.latency_ms for job in jobs if job["tier"] == tier.value]
        )
        for tier, profile in profiles.items()
    }
    result["inference_degradation_scope"] = (
        "relative to isolated frozen profile; descriptive, not causal contention"
    )
    return result


def run_matrix(path: Path) -> Path:
    matrix, resolved = freeze_matrix(path)
    output = matrix.output_dir
    study_sha256 = hash_file(output / "matrix.resolved.json")
    lock = output / "execution.lock"
    # Exclusive creation, never silently steal a stale/live owner lock.
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "study_sha256": study_sha256}, stream)
    statuses = []
    try:
        for subrun in resolved["subruns"]:
            subrun_id = subrun["id"]
            directory = output / "subruns" / subrun_id
            if subrun["status"] == "skipped":
                statuses.append({**subrun, "status": "skipped"})
                continue
            if (directory / "complete.json").exists():
                # Integrity failures are fatal, not invitations to overwrite evidence.
                saved = read_completed(directory, study_sha256, subrun_id)
                statuses.append({**subrun, "status": "complete", "run": saved["run"]})
                continue
            config = load_config(output / "configs" / f"{subrun['case']}.yaml")
            directory.mkdir(parents=True, exist_ok=True)
            attempt_id = uuid.uuid4().hex
            print(f"START {subrun_id}", flush=True)
            started = time.perf_counter_ns()
            run: Path | None = None
            try:
                with (
                    open(os.devnull, "w") as sink,
                    contextlib.redirect_stdout(sink),
                    contextlib.redirect_stderr(sink),
                ):
                    if subrun["component"]:
                        run, metrics = run_component(
                            config,
                            subrun["component"],
                            subrun["component_level"],
                            matrix.warmup_forwards_per_tier,
                        )
                    else:
                        run = run_concurrent(
                            config, warmup_forwards_per_tier=matrix.warmup_forwards_per_tier
                        )
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                if not subrun["component"]:
                    metrics = inspect_run(config, run, matrix)
                verify_hashes(matrix.project_root, resolved["inputs"])
                if source_inventory(matrix.project_root) != resolved["source_hashes"]:
                    raise ValueError("runtime source changed during subrun")
                metrics["total_invocation_wall_ms"] = wall_ms
                metrics["total_invocation_scope"] = (
                    "setup, loading, warmup, stream and finalization; excludes analysis"
                )
                metrics_file = f"attempts/{attempt_id}/metrics.json"
                write_immutable(directory / metrics_file, canonical(metrics))
                completed = {
                    "schema_version": 1,
                    "kind": "component" if subrun["component"] else "pipeline",
                    "study_sha256": study_sha256,
                    "subrun_id": subrun_id,
                    "run": str(run.resolve()),
                    "run_hashes": {
                        name: hash_file(run / name)
                        for name in (COMPONENT_FILES if subrun["component"] else RUN_FILES)
                    },
                    "metrics_file": metrics_file,
                    "metrics_sha256": hash_file(directory / metrics_file),
                }
                write_immutable(directory / "complete.json", canonical(completed))
                statuses.append({**subrun, "status": "complete", "run": str(run)})
                print(f"COMPLETE {subrun_id}: {metrics['occurrences']} candidates", flush=True)
            except Exception as error:
                # Each failed attempt remains immutable; a retry gets a new runtime run.
                failure = {
                    **subrun,
                    "status": "failed",
                    "attempt": attempt_id,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "run": str(run) if run else None,
                }
                write_immutable(directory / f"failed_{attempt_id}.json", canonical(failure))
                statuses.append(failure)
                print(f"FAILED {subrun_id}: {error}", flush=True)
        report = {
            "schema_version": 1,
            "study_sha256": study_sha256,
            "subruns": statuses,
            "test_evaluated": matrix.partition == "test",
        }
        content = canonical(report)
        report_path = (
            output / "attempts" / f"matrix_{hashlib.sha256(content).hexdigest()[:16]}.json"
        )
        write_immutable(report_path, content)
        return report_path
    finally:
        lock.unlink()


def summarize_study(directory: Path) -> Path:
    resolved_path = directory / "matrix.resolved.json"
    study_sha256 = hash_file(resolved_path)
    resolved = json.loads(resolved_path.read_bytes())
    subruns = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for subrun in resolved["subruns"]:
        folder = directory / "subruns" / subrun["id"]
        item = dict(subrun)
        item["failed_attempts"] = [str(p) for p in sorted(folder.glob("failed_*.json"))]
        if subrun["status"] == "skipped":
            pass
        elif (folder / "complete.json").exists():
            completed = read_completed(folder, study_sha256, subrun["id"])
            item.update(
                status="complete",
                metrics=json.loads((folder / completed["metrics_file"]).read_bytes()),
            )
            groups.setdefault(subrun["case"], []).append(item["metrics"])
        else:
            failures = sorted(folder.glob("failed_*.json"))
            item.update(
                status="failed" if failures else "pending",
                reason="recorded failed attempts" if failures else "not completed",
                attempts=[json.loads(p.read_bytes()) for p in failures],
            )
        subruns.append(item)
    summary = {
        "schema_version": 1,
        "study_sha256": study_sha256,
        "design": "curated_not_full_factorial",
        "purpose": resolved["matrix"]["purpose"],
        "cases": resolved["cases"],
        "counts": {
            status: sum(r["status"] == status for r in subruns)
            for status in ("complete", "failed", "skipped", "pending")
        },
        "subruns": subruns,
        "timing_by_case": {
            key: {
                "repetitions": len(values),
                "throughput_candidates_per_s": distribution(
                    [v["throughput_candidates_per_s"] for v in values]
                ),
                "p95_end_to_end_ms": distribution(
                    [v["latency_ms"]["end_to_end_ms"]["p95"] for v in values]
                ),
            }
            for key, values in groups.items()
        },
        "statistical_scope": (
            "descriptive timing distributions; science stays per repetition; no pooling"
        ),
        "test_evaluated": resolved["matrix"]["partition"] == "test",
    }
    content = canonical(summary)
    path = directory / "summaries" / f"summary_{hashlib.sha256(content).hexdigest()[:16]}.json"
    write_immutable(path, content)
    return path


def plot_summary(path: Path) -> Path:
    """Initial systems QA plots; scientific plots require a completed scientific matrix."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    value = json.loads(path.read_bytes())
    if value.get("purpose") in {"unique_test", "unique_validation"}:
        from edge_triage.experiments.figures import scientific_plots

        return scientific_plots(path)
    if value.get("schema_version") != 1 or "timing_by_case" not in value:
        raise ValueError("unsupported experiment summary schema")
    items = value["timing_by_case"]
    if not items:
        raise ValueError("no completed repetitions to plot")
    directory = path.parent / f"plots_{hash_file(path)[:16]}"
    result: dict[str, Any] = {
        "schema_version": 1,
        "summary_sha256": hash_file(path),
        "plots": {},
        "scope": "systems QA; full nine scientific plots pending held-out protocol",
    }
    for field, unit in (
        ("throughput_candidates_per_s", "candidates / second"),
        ("p95_end_to_end_ms", "P95 end-to-end latency (ms)"),
    ):
        fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.45), 5), layout="constrained")
        labels = list(items)
        centers = [items[k][field]["mean"] for k in labels]
        ax.bar(range(len(labels)), centers, color="#3577a8")
        for i, key in enumerate(labels):
            values = [
                r["metrics"]["throughput_candidates_per_s"]
                if field.startswith("throughput")
                else r["metrics"]["latency_ms"]["end_to_end_ms"]["p95"]
                for r in value["subruns"]
                if r["case"] == key and r["status"] == "complete"
            ]
            ax.scatter([i] * len(values), values, color="#212121", s=16)
        ax.set_xticks(
            range(len(labels)),
            [f"{k}\nn={items[k]['repetitions']} runs" for k in labels],
            rotation=70,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(unit)
        ax.set_title(f"{value['purpose']}: means and individual repetitions")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, metadata={"Software": "edge-triage"})
        plt.close(fig)
        destination = directory / f"{field}.png"
        write_immutable(destination, buffer.getvalue())
        result["plots"][destination.name] = hash_file(destination)
    destination = directory / "index.json"
    write_immutable(destination, canonical(result))
    return destination
