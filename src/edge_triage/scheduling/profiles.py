"""Prepare bounded policy inputs once, outside the decision critical path."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from edge_triage.calibration.profiling import load_profiles
from edge_triage.config import AppConfig
from edge_triage.contracts import Contract, ModelTier, PositiveCount
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest, read_index


class SchedulerProfile(Contract):
    tier: ModelTier
    profile_id: str
    checkpoint_id: str
    latency_ms: float = Field(gt=0)
    latency_p95_ms: float = Field(gt=0)
    memory_bytes: PositiveCount
    memory_basis: str


def scheduler_profiles(config: AppConfig) -> dict[ModelTier, SchedulerProfile]:
    settings = config.scheduler
    if not (
        settings.profiles
        and settings.profiles_sha256
        and settings.hardware_id
        and config.models.checkpoint_index
    ):
        raise ValueError("scheduler requires pinned profiles, hardware and checkpoint family")
    if settings.device != config.training.device or config.training.cpu_threads != 1:
        raise ValueError("scheduler profile device/CPU thread mismatch")
    family = read_index(config.models.checkpoint_index)
    manifests = {}
    for entry in family["checkpoints"].values():
        path = Path(entry["directory"]) / "checkpoint_manifest.json"
        if hash_file(path) != entry["manifest_sha256"]:
            raise ValueError("checkpoint manifest hash mismatch")
        manifests[entry["checkpoint_id"]] = (
            CheckpointManifest.model_validate_json(path.read_bytes()),
            entry["manifest_sha256"],
        )
    bundle = load_profiles(settings.profiles, settings.profiles_sha256, manifests)
    if bundle.family_index_sha256 != hash_file(config.models.checkpoint_index):
        raise ValueError("profile family mismatch")
    selected = {}
    for tier in config.models.tier_portfolio:
        matches = [
            p
            for p in bundle.profiles
            if p.tier == tier
            and p.device == settings.device
            and p.hardware_id == settings.hardware_id
            and p.batch_size == 1
            and p.concurrency == 1
            and p.precision_mode == "fp32_tf32_off"
            and p.input_shape
            == (
                len(config.preprocessing.selected_channels),
                config.preprocessing.crop_size,
                config.preprocessing.crop_size,
            )
            and p.checkpoint_id == family["checkpoints"][tier.value]["checkpoint_id"]
        ]
        if len(matches) != 1 or not matches[0].stable or matches[0].failures:
            raise ValueError(f"no unique stable compatible profile for {tier.value}")
        row = matches[0]
        # Full process RSS is deliberately conservative; never invent an activation-only cost.
        memory = (
            row.host_rss_sampled_peak_bytes
            if settings.device == "cpu"
            else row.cuda_reserved_peak_bytes
        )
        if memory is None or memory <= 0:
            raise ValueError("profile memory unavailable")
        selected[tier] = SchedulerProfile(
            tier=tier,
            profile_id=row.profile_id,
            checkpoint_id=row.checkpoint_id,
            latency_ms=row.latency_p50_ms,
            latency_p95_ms=row.latency_p95_ms,
            memory_bytes=memory,
            memory_basis="whole_process_rss" if settings.device == "cpu" else "cuda_reserved_peak",
        )
    return selected
