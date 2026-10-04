"""Synchronized wall-clock profiling and scheduler-readable profile contracts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import psutil
import torch
from pydantic import Field, model_validator

from edge_triage.contracts import Contract, ModelProfile
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest


class Latency(Contract):
    mean_ms: float = Field(gt=0, allow_inf_nan=False)
    median_ms: float = Field(gt=0, allow_inf_nan=False)
    p90_ms: float = Field(gt=0, allow_inf_nan=False)
    p95_ms: float = Field(gt=0, allow_inf_nan=False)
    p99_ms: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.median_ms <= self.p90_ms <= self.p95_ms <= self.p99_ms:
            raise ValueError("unordered latency quantiles")
        return self


class MeasuredProfile(ModelProfile):
    schema_version: Literal[1] = 1
    hardware_id: str
    precision_mode: Literal["fp32_tf32_off"] = "fp32_tf32_off"
    checkpoint_manifest_sha256: str
    split_sha256: str
    preprocessing_sha256: str
    parameter_count: int = Field(gt=0)
    checkpoint_bytes: int = Field(gt=0)
    load_ms: float = Field(ge=0)
    warmup: int = Field(gt=0)
    rounds: int = Field(ge=2)
    components: dict[str, Latency]
    throughput_candidates_s: float = Field(gt=0)
    amortized_ms_per_candidate: float = Field(gt=0)
    batch_completion_median_ms: float = Field(gt=0)
    cuda_allocated_peak_bytes: int | None
    cuda_reserved_peak_bytes: int | None
    host_rss_before_bytes: int
    host_rss_after_bytes: int
    host_rss_sampled_peak_bytes: int
    round_medians: tuple[float, ...]
    stability_ratio: float = Field(ge=1)
    stable: bool
    raw_samples_path: str
    raw_samples_sha256: str
    failures: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.hardware_id,
            self.device,
            self.tier,
            self.checkpoint_id,
            self.input_shape,
            self.batch_size,
            self.precision_mode,
        )


class ProfileBundle(Contract):
    schema_version: Literal[1] = 1
    config_sha256: str
    code_sha256: str
    family_index_sha256: str
    profiles: tuple[MeasuredProfile, ...]
    environment: dict[str, Any]
    methodology: dict[str, Any]
    overhead: dict[str, Any]

    @model_validator(mode="after")
    def unique(self) -> Self:
        if not self.profiles or len({p.key for p in self.profiles}) != len(self.profiles):
            raise ValueError("empty or duplicate profile keys")
        return self


def load_profiles(
    path: Path, expected_sha256: str, manifests: dict[str, tuple[CheckpointManifest, str]]
) -> ProfileBundle:
    if hash_file(path) != expected_sha256:
        raise ValueError("profile hash mismatch")
    bundle = ProfileBundle.model_validate_json(path.read_bytes())
    for row in bundle.profiles:
        manifest, digest = manifests[row.checkpoint_id]
        if (
            row.checkpoint_manifest_sha256 != digest
            or row.tier != manifest.tier
            or row.split_sha256 != manifest.split_sha256
            or row.preprocessing_sha256 != manifest.preprocessing_sha256
            or row.parameter_count != manifest.parameter_count
        ):
            raise ValueError("incompatible checkpoint/profile")
        if hash_file(Path(row.raw_samples_path)) != row.raw_samples_sha256:
            raise ValueError("raw timing sample hash mismatch")
    return bundle


def synchronize(device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(operation: Callable[[], Any], device: Any, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter_ns()
        operation()
        synchronize(device)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return samples


def summarize(samples: list[float]) -> Latency:
    return Latency(
        mean_ms=float(np.mean(samples)),
        median_ms=float(np.median(samples)),
        p90_ms=float(np.percentile(samples, 90)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=float(np.percentile(samples, 99)),
    )


def hardware_identity() -> tuple[str, dict[str, Any]]:
    identity = {
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else None,
        "os": platform.platform(),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16], identity


def machine_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cpu_percent_sample": psutil.cpu_percent(interval=0.1),
        "available_ram_bytes": psutil.virtual_memory().available,
    }
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pstate,utilization.gpu,"
                "memory.used,temperature.gpu,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        result["nvidia_smi_fields"] = (
            "name,driver_version,pstate,utilization.gpu_pct,memory.used_MiB,temperature.C,"
            "clocks.sm_MHz,clocks.mem_MHz"
        )
        result["nvidia_smi"] = probe.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        result["nvidia_smi_unavailable"] = type(error).__name__
    return result
