"""Hash-bound measured concurrency limits; unknown conditions conservatively use one."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from edge_triage.config import AppConfig
from edge_triage.contracts import SHA256, Contract, PositiveCount
from edge_triage.health import hash_file


class ContentionProfile(Contract):
    schema_version: Literal[1] = 1
    hardware_id: str
    device: Literal["cpu", "cuda"]
    family_sha256: SHA256
    profiles_sha256: SHA256
    cpu_threads: Literal[1] = 1
    input_shape: tuple[int, int, int] = (3, 30, 30)
    precision: Literal["fp32_tf32_off"] = "fp32_tf32_off"
    benchmark_sha256: SHA256
    criterion: str
    caps: dict[str, PositiveCount] = Field(default_factory=dict)


def load_contention_cap(config: AppConfig) -> int:
    settings = config.scheduler
    if settings.contention_profile is None or settings.contention_sha256 is None:
        raise ValueError("contention profile requires path and checksum")
    if hash_file(settings.contention_profile) != settings.contention_sha256:
        raise ValueError("contention profile checksum mismatch")
    profile = ContentionProfile.model_validate_json(settings.contention_profile.read_bytes())
    if (
        profile.hardware_id != settings.hardware_id
        or profile.device != settings.device
        or config.models.checkpoint_index is None
        or profile.family_sha256 != hash_file(config.models.checkpoint_index)
        or profile.profiles_sha256 != settings.profiles_sha256
        or profile.cpu_threads != config.training.cpu_threads
        or profile.input_shape
        != (
            len(config.preprocessing.selected_channels),
            config.preprocessing.crop_size,
            config.preprocessing.crop_size,
        )
    ):
        raise ValueError("incompatible contention profile")
    return profile.caps.get("+".join(t.value for t in config.models.tier_portfolio), 1)
