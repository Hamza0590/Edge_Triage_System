"""Validation-fitted, hash-bound routing definitions (no runtime labels)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from edge_triage.contracts import SHA256, Contract, ModelTier, Probability
from edge_triage.health import hash_file


class TierUncertainty(Contract):
    tier: ModelTier
    checkpoint_id: str
    calibration_artifact_id: str
    calibration_sha256: SHA256
    probability_mode: Literal["calibrated", "uncalibrated"]
    temperature: float = Field(gt=0)
    lower: Probability
    upper: Probability
    std_threshold: float = Field(ge=0, le=0.5)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("ambiguous interval is reversed")
        return self


class UncertaintyArtifact(Contract):
    schema_version: Literal[1] = 1
    artifact_id: str
    policy_version: Literal["selective_mc_v1"] = "selective_mc_v1"
    fit_partition: Literal["validation"] = "validation"
    split_sha256: SHA256
    preprocessing_sha256: SHA256
    family_sha256: SHA256
    calibration_index_sha256: SHA256
    objective: str
    passes: Literal[5, 10, 20]
    tiers: tuple[TierUncertainty, ...]

    @model_validator(mode="after")
    def unique(self) -> Self:
        if not self.tiers or len({t.tier for t in self.tiers}) != len(self.tiers):
            raise ValueError("unique tier routing definitions required")
        return self


def load_uncertainty(path: Path, digest: str) -> UncertaintyArtifact:
    if hash_file(path) != digest:
        raise ValueError("uncertainty artifact hash mismatch")
    return UncertaintyArtifact.model_validate_json(path.read_bytes())
