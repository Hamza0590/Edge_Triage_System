"""Immutable validation-only system arrival traces, independent of execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from edge_triage.config import AppConfig, WorkloadConfig
from edge_triage.contracts import Contract, Count, Identifier, NonNegative
from edge_triage.data.artifacts import canonical, frozen_split, load_raw, write_immutable
from edge_triage.health import hash_file

PROFILES = ("low", "moderate", "high", "burst", "alternating", "backlog")


class TraceRecord(Contract):
    trace_position: Count
    candidate_id: Identifier
    source_candidate_id: Identifier
    dataset_index: Count
    relative_arrival_seconds: NonNegative
    workload_phase: str
    seed: Count
    repetition: Count
    independent_classification_sample: Literal[False] = False


def generate(rows: list[dict[str, Any]], spec: WorkloadConfig) -> list[TraceRecord]:
    if not rows or any(r["split"] != "validation" for r in rows):
        raise ValueError("system workload source must be nonempty validation rows")
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source rows")
    rng = np.random.default_rng(spec.seed)
    result = []
    elapsed = 0.0
    for position in range(spec.count):
        row = rows[position % len(rows)]
        repetition = position // len(rows)
        phase = spec.profile
        if phase == "burst":
            phase = "high" if spec.count // 3 <= position < 2 * spec.count // 3 else "low"
        elif phase == "alternating":
            phase = "low" if (position // max(1, spec.count // 6)) % 2 == 0 else "high"
        if spec.profile != "backlog" and position:
            rate = getattr(spec, f"{phase}_rate")
            elapsed += (
                1 / rate if spec.arrival_process == "periodic" else float(rng.exponential(1 / rate))
            )
        result.append(
            TraceRecord(
                trace_position=position,
                candidate_id=f"{row['candidate_id']}:occurrence-{position}",
                source_candidate_id=row["candidate_id"],
                dataset_index=row["index_no"],
                relative_arrival_seconds=elapsed,
                workload_phase=phase,
                seed=spec.seed,
                repetition=repetition,
            )
        )
    return result


def persist_trace(config: AppConfig, spec: WorkloadConfig) -> Path:
    _, labels, _ = load_raw(config)
    rows, split_hash = frozen_split(config, labels)
    records = generate([r for r in rows if r["split"] == "validation"], spec)
    settings = spec.model_dump(mode="json", exclude={"trace"})
    value = {
        "schema_version": 1,
        "partition": "validation",
        "split_sha256": split_hash,
        "settings": settings,
        "records": [r.model_dump(mode="json") for r in records],
        "purpose": "SYSTEMS_LOAD_ONLY_NO_CLASSIFICATION_INFERENCE",
    }
    content = canonical(value)
    digest = hashlib.sha256(content).hexdigest()
    path = (
        config.paths.artifacts_dir
        / "traces"
        / f"{spec.profile}_{spec.arrival_process}_{digest[:16]}.json"
    )
    write_immutable(path, content)
    write_immutable(path.with_suffix(".sha256"), (digest + "\n").encode())
    return path


def load_trace(path: Path, split_sha256: str) -> list[TraceRecord]:
    if hash_file(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("trace checksum mismatch")
    value = json.loads(path.read_text())
    if (
        value["schema_version"] != 1
        or value["partition"] != "validation"
        or value["split_sha256"] != split_sha256
    ):
        raise ValueError("trace identity mismatch")
    records = [TraceRecord.model_validate(r) for r in value["records"]]
    if not records or len({r.candidate_id for r in records}) != len(records):
        raise ValueError("empty trace or duplicate occurrence ID")
    if [r.trace_position for r in records] != list(range(len(records))):
        raise ValueError("trace positions invalid")
    if any(
        b.relative_arrival_seconds < a.relative_arrival_seconds
        for a, b in zip(records, records[1:])
    ):
        raise ValueError("trace arrivals out of order")
    return records


def generate_suite(config: AppConfig) -> Path:
    entries = []
    for profile in PROFILES:
        for process in ("periodic", "poisson"):
            spec = WorkloadConfig.model_validate(
                {**config.workload.model_dump(), "profile": profile, "arrival_process": process}
            )
            path = persist_trace(config, spec)
            entries.append(
                {
                    "profile": profile,
                    "process": process,
                    "path": str(path),
                    "sha256": hash_file(path),
                }
            )
    content = canonical({"schema_version": 1, "traces": entries})
    digest = hashlib.sha256(content).hexdigest()
    path = config.paths.artifacts_dir / "traces" / f"suite_{digest[:16]}.json"
    write_immutable(path, content)
    return path
