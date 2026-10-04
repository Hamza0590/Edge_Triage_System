"""Read-only dataset integrity and observable environment diagnostics."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from edge_triage.config import AppConfig
from edge_triage.contracts import ArtifactHash


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hashes(config: AppConfig) -> tuple[ArtifactHash, ...]:
    """Stream bytes in bounded chunks; never load or alter the NumPy array."""
    artifacts = []
    for name, path, expected in (
        ("RAW_IMAGES", config.dataset.images, config.dataset.images_md5),
        ("RAW_LABELS", config.dataset.labels, config.dataset.labels_md5),
    ):
        actual = hash_file(path, "md5")
        if actual != expected:
            raise ValueError(f"{name} MD5 mismatch: expected {expected}, observed {actual}")
        artifacts.append(
            ArtifactHash(artifact_id=name, path=str(path), algorithm="md5", digest=actual)
        )
    return tuple(artifacts)


def nvidia_devices() -> list[dict[str, Any]]:
    """Observe NVIDIA hardware even when PyTorch is not installed."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return [
            {
                "name": row[0].strip(),
                "memory_bytes": int(row[1].strip()) * 1024 * 1024,
                "source": "nvidia-smi",
            }
            for row in csv.reader(result.stdout.splitlines())
            if row
        ]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []


def environment_info() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for dependency in ("edge-triage", "pydantic", "PyYAML", "psutil", "torch", "numpy", "pyarrow"):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    info: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "dependencies": versions,
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu": platform.processor(),
        "ram_bytes": psutil.virtual_memory().total,
        "cuda_available": False,
        "cuda_version": None,
        "gpus": [],
        "torch_error": None,
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_properties(index).name,
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "source": "torch",
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch_error"] = "PyTorch unavailable; optional for foundation"
    except Exception as error:
        info["torch_error"] = f"PyTorch/CUDA probe failed: {type(error).__name__}"
    if not info["gpus"]:
        info["gpus"] = nvidia_devices()
    return info


def git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return commit.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def doctor(config: AppConfig) -> dict[str, Any]:
    from edge_triage.provenance import load_provenance

    if not config.paths.project_root.is_dir():
        raise ValueError("project_root is not an existing directory")
    directories = {}
    for name in ("runs_dir", "artifacts_dir", "logs_dir"):
        path = getattr(config.paths, name)
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            raise ValueError(f"{name} does not have a writable directory ancestor")
        directories[name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "writable_ancestor": str(ancestor),
        }
    sources = load_provenance(config.paths.provenance_file)
    source_ids = {source.id for source in sources}
    if not set(config.experiment.provenance_ids) <= source_ids:
        raise ValueError("unknown experiment provenance ID")
    hashes = dataset_hashes(config)
    environment = environment_info()
    if config.training.device == "cuda" and not environment["cuda_available"]:
        raise ValueError("CUDA was requested but is unavailable")
    return {
        "ok": True,
        "config_hash": config.config_hash,
        "directories": directories,
        "dataset_hashes": [artifact.model_dump(mode="json") for artifact in hashes],
        "environment": environment,
        "provenance_records": len(sources),
    }
