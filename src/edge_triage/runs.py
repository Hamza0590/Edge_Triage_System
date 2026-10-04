"""Single-owner run lifecycle with immutable final files and appended corrections."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from edge_triage.config import AppConfig
from edge_triage.contracts import ArtifactHash, RunManifest, utc_now
from edge_triage.event_logging import EventLogger, safe_text
from edge_triage.health import dataset_hashes, environment_info, git_state, hash_file
from edge_triage.provenance import load_provenance


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _optional_hash(path: Path | None, artifact_id: str) -> ArtifactHash | None:
    if path is None:
        return None
    return ArtifactHash(
        artifact_id=artifact_id, path=str(path), algorithm="sha256", digest=hash_file(path)
    )


class RunManager:
    """Own one run and its logger; finalization is allowed exactly once.

    Future executors share this owner/logger, rather than independently writing
    the same manifest. Immutability is enforced by APIs, not filesystem ACLs.
    """

    def __init__(self, config: AppConfig) -> None:
        # Verify inputs before creating the run so invalid datasets leave no partial run.
        hashes = dataset_hashes(config)
        registry = load_provenance(config.paths.provenance_file)
        if not set(config.experiment.provenance_ids) <= {entry.id for entry in registry}:
            raise ValueError("unknown experiment provenance ID")
        split = _optional_hash(config.split.manifest, "split")
        preprocessing = _optional_hash(config.preprocessing.artifact, "preprocessing")
        threshold = _optional_hash(config.threshold.artifact, "threshold")
        trace = _optional_hash(config.workload.trace, "arrival_trace")
        started = utc_now()
        run_id = started.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
        commit, dirty = git_state(config.paths.project_root)
        self.manifest = RunManifest(
            run_id=run_id,
            purpose=config.experiment.purpose,
            started_utc=started,
            config_hash=config.config_hash,
            dataset_hashes=hashes,
            split_hash=split,
            preprocessing_hash=preprocessing,
            threshold_hashes=(threshold,) if threshold else (),
            arrival_trace_hash=trace,
            git_commit=commit,
            git_dirty=dirty,
            environment=environment_info(),
            seeds={
                name: getattr(config, name).seed
                for name in (
                    "split",
                    "training",
                    "calibration",
                    "uncertainty",
                    "workload",
                    "experiment",
                )
            },
            provenance_ids=config.experiment.provenance_ids,
        )
        self.directory = config.paths.runs_dir / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "metrics").mkdir()
        (self.directory / "artifacts").mkdir()
        _atomic_text(self.directory / "config.resolved.yaml", config.resolved_yaml())
        _atomic_text(self.directory / "manifest.json", self.manifest.model_dump_json(indent=2))
        self.logger = EventLogger(self.directory / "events.jsonl", run_id, config.config_hash)
        self._lock = threading.RLock()
        self.logger.emit(
            module="runs",
            event_type="run.started",
            reason_code="RUN_CREATED",
            message=config.experiment.purpose,
        )

    def register_threshold(self, path: Path) -> None:
        """Attach a generated threshold artifact before finalization."""
        with self._lock:
            if self.manifest.status != "running":
                raise RuntimeError("finalized run is immutable")
            artifact = _optional_hash(path, "threshold")
            self.manifest = RunManifest.model_validate(
                {
                    **self.manifest.model_dump(),
                    "threshold_hashes": (*self.manifest.threshold_hashes, artifact),
                }
            )
            _atomic_text(self.directory / "manifest.json", self.manifest.model_dump_json(indent=2))

    def register_checkpoint(self, path: Path, checkpoint_id: str) -> None:
        """Link an immutable checkpoint manifest before sealing this run."""
        with self._lock:
            if self.manifest.status != "running":
                raise RuntimeError("finalized run is immutable")
            artifact = _optional_hash(path, checkpoint_id)
            self.manifest = RunManifest.model_validate(
                {
                    **self.manifest.model_dump(),
                    "checkpoint_hashes": (*self.manifest.checkpoint_hashes, artifact),
                }
            )
            _atomic_text(self.directory / "manifest.json", self.manifest.model_dump_json(indent=2))

    def finalize(self, status: Literal["completed", "failed"] = "completed") -> RunManifest:
        with self._lock:
            disk_manifest = RunManifest.model_validate_json(
                (self.directory / "manifest.json").read_text(encoding="utf-8")
            )
            if disk_manifest.status != "running":
                raise RuntimeError("run already finalized; append an explicit correction instead")
            final = RunManifest.model_validate(
                {
                    **self.manifest.model_dump(),
                    "status": status,
                    "ended_utc": utc_now(),
                }
            )
            self.logger.emit(
                module="runs",
                event_type=f"run.{status}",
                reason_code=f"RUN_{status.upper()}",
                message=f"Run {status}",
            )
            self.logger.close()
            _atomic_text(self.directory / "manifest.json", final.model_dump_json(indent=2))
            self.manifest = final
            return final

    def __enter__(self) -> RunManager:
        return self

    def __exit__(self, exc_type: Any, exception: Any, traceback: Any) -> None:
        if self.manifest.status != "running":
            return
        if exception is not None:
            self.logger.emit(
                module="runs",
                event_type="error.run_failed",
                reason_code="UNHANDLED",
                level="ERROR",
                message="Run failed",
                exception=exception,
            )
        self.finalize("failed" if exception is not None else "completed")


def append_correction(directory: Path, *, author: str, reason: str, correction: str) -> None:
    """Append an audit note without modifying finalized artifacts or manifest."""
    manifest = RunManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.status == "running":
        raise ValueError("corrections apply only to finalized runs")
    if not all(value.strip() for value in (author, reason, correction)):
        raise ValueError("author, reason and correction are required")
    record = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "timestamp_utc": utc_now().isoformat(),
        "author": safe_text(author, 256),
        "reason": safe_text(reason),
        "correction": safe_text(correction),
        "manifest_sha256": hash_file(directory / "manifest.json"),
    }
    with (directory / "corrections.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
