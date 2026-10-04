"""Versioned, explicit engineering constraints; recall is a reference, not a guarantee."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from edge_triage.contracts import (
    SHA256,
    Contract,
    Count,
    ModelTier,
    NonNegative,
    Percentage,
    PositiveCount,
    Probability,
)


class SchedulerConstraints(Contract):
    target_survival_recall: Probability = 0.95
    latency_slo_ms: NonNegative | None = None
    queue_wait_slo_ms: NonNegative | None = None
    memory_margin_bytes: Count = 268435456
    cpu_low: Percentage = 50
    cpu_high: Percentage = 90
    gpu_low: Percentage = 50
    gpu_high: Percentage = 90
    queue_low: Count = 1
    queue_high: PositiveCount = 6
    per_tier_caps: dict[ModelTier, PositiveCount] = Field(
        default_factory=lambda: {tier: 1 for tier in ModelTier}
    )
    max_total_active: PositiveCount = 4
    contention_ratio: float = Field(default=2.0, gt=1)
    min_latency_samples: PositiveCount = 5
    hysteresis_s: NonNegative = 0.5
    cooldown_s: NonNegative = 1
    max_snapshot_age_s: float = Field(default=0.5, gt=0)
    max_gpu_age_s: float = Field(default=2.5, gt=0)
    preserve_escalation_target: bool = True

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not (
            self.cpu_low < self.cpu_high
            and self.gpu_low < self.gpu_high
            and self.queue_low < self.queue_high
        ):
            raise ValueError("low watermarks must be below high watermarks")
        if set(self.per_tier_caps) != set(ModelTier):
            raise ValueError("specify caps for all tiers; portfolio controls availability")
        return self


class SchedulerConfig(Contract):
    contention_profile: Path | None = None
    contention_sha256: SHA256 | None = None
    profiles: Path | None = None
    profiles_sha256: SHA256 | None = None
    hardware_id: str | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    constraints: SchedulerConstraints = Field(default_factory=SchedulerConstraints)
