"""Immutable fate operating points, with exact system and input identity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from edge_triage.config import AppConfig, omit_legacy_defaults
from edge_triage.contracts import SHA256, Contract, ModelTier, Probability
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.runtime.interfaces import ThresholdDecision


class FateOperatingPoint(Contract):
    # None explicitly disables discard, including for exactly-zero probabilities.
    max_p_real_for_discard: float | None = Field(default=None, ge=0, lt=0.5)
    min_p_real_for_transmit: float = Field(default=0.5, ge=0.5, le=1)
    max_uncertainty_for_discard: float = Field(default=0, ge=0, le=0.5)


class TierFateDefinition(Contract):
    tier: ModelTier
    checkpoint_id: str
    probability_mode: Literal["calibrated", "uncalibrated"]
    calibration_artifact_id: str | None = None
    discard_supported: bool = True
    operating_point: FateOperatingPoint

    @model_validator(mode="after")
    def calibration_required(self) -> Self:
        if (self.probability_mode == "calibrated") != (self.calibration_artifact_id is not None):
            raise ValueError("calibrated mode requires exactly one calibration identity")
        return self


class FateThresholdArtifact(Contract):
    schema_version: Literal[1] = 1
    algorithm_version: Literal["recall_fate_v1"] = "recall_fate_v1"
    artifact_id: str
    fit_partition: Literal["validation"] = "validation"
    threshold_policy: Literal["common", "per_tier"]
    system_sha256: SHA256
    family_sha256: SHA256
    split_sha256: SHA256
    preprocessing_sha256: SHA256
    calibration_index_sha256: SHA256
    validation_records_sha256: SHA256
    target_survival_recall: Probability
    validation_survival_recall: Probability
    monotonic_discard: bool = True
    objective: str
    validation_metrics: dict[str, Any]
    tiers: tuple[TierFateDefinition, ...]

    @model_validator(mode="after")
    def validate_definitions(self) -> Self:
        tiers = [entry.tier for entry in self.tiers]
        if not tiers or len(set(tiers)) != len(tiers):
            raise ValueError("unique nonempty fate tiers required")
        if tiers != [tier for tier in ModelTier if tier in tiers]:
            raise ValueError("fate tiers must be ordered weakest to strongest")
        if self.validation_survival_recall < self.target_survival_recall:
            raise ValueError("cannot freeze an operating point below target recall")
        points = [entry.operating_point for entry in self.tiers]
        if self.threshold_policy == "common" and any(p != points[0] for p in points):
            raise ValueError("common thresholds must be identical across tiers")
        cutoffs = [
            -1.0 if p.max_p_real_for_discard is None else p.max_p_real_for_discard for p in points
        ]
        if self.monotonic_discard and cutoffs != sorted(cutoffs):
            raise ValueError("weaker tiers must require at least as strong discard evidence")
        return self


def system_identity(config: AppConfig) -> str:
    """Bind all behavior, excluding output locations, partition and self-reference.

    Test partition may consume a frozen point; it must never fit one. Trace identity
    stays bound: a different scheduler workload requires separate validation.
    """
    raw = config.model_dump(mode="json")
    raw["pipeline"].update(promotion=None, promotion_sha256=None)
    omit_legacy_defaults(raw)
    raw["threshold"].update(artifact=None, artifact_sha256=None)
    raw["pipeline"]["partition"] = "validation"
    raw["experiment"]["name"] = "fate_system_identity"
    for name in ("runs_dir", "artifacts_dir", "logs_dir"):
        raw["paths"].pop(name)
    raw["workload"]["trace_content_sha256"] = (
        hash_file(config.workload.trace) if config.workload.trace is not None else None
    )
    return hashlib.sha256(canonical(raw)).hexdigest()


def load_fate_artifact(path: Path, digest: str, config: AppConfig) -> FateThresholdArtifact:
    if hash_file(path) != digest:
        raise ValueError("fate threshold artifact hash mismatch")
    artifact = FateThresholdArtifact.model_validate_json(path.read_bytes())
    if path.read_bytes() != canonical(artifact.model_dump(mode="json")):
        raise ValueError("fate artifact must use canonical immutable encoding")
    references = {
        "family_sha256": config.models.checkpoint_index,
        "split_sha256": config.split.manifest,
        "preprocessing_sha256": config.preprocessing.artifact,
        "calibration_index_sha256": config.calibration.artifact_index,
    }
    if any(
        path is None or hash_file(path) != getattr(artifact, key)
        for key, path in references.items()
    ):
        raise ValueError("incompatible fate input artifacts")
    identity_config = config
    if config.pipeline.partition == "test":
        from edge_triage.experiments.promotion import verify_promotion

        identity_config = verify_promotion(config)
    if (
        artifact.system_sha256 != system_identity(identity_config)
        or tuple(t.tier for t in artifact.tiers) != config.models.tier_portfolio
        or artifact.threshold_policy != config.threshold.policy
        or artifact.target_survival_recall != config.fate.target_survival_recall
    ):
        raise ValueError("incompatible fate system configuration")
    return artifact


class PerTierThresholdPolicy:
    def __init__(self, artifact: FateThresholdArtifact) -> None:
        self.artifact = artifact
        self._definition = canonical(artifact.model_dump(mode="json"))
        self.digest = hashlib.sha256(self._definition).hexdigest()

    def definition(self) -> bytes:
        return self._definition

    def persist(self, path: Path) -> None:
        write_immutable(path, self.definition())

    def resolve(self, tier: ModelTier) -> ThresholdDecision:
        spec = next((entry for entry in self.artifact.tiers if entry.tier == tier), None)
        if spec is None:
            raise ValueError("tier absent from frozen fate artifact")
        return ThresholdDecision(
            p_real_threshold=0.5,
            artifact_id=self.artifact.artifact_id,
            artifact_sha256=self.digest,
            checkpoint_id=spec.checkpoint_id,
            trusted_tier=spec.tier,
            discard_supported=spec.discard_supported,
            probability_mode=spec.probability_mode,
            calibration_artifact_id=spec.calibration_artifact_id,
            **spec.operating_point.model_dump(),
        )


class CommonThresholdPolicy(PerTierThresholdPolicy):
    """Scientific common fate threshold, distinct from the legacy smoke classifier."""

    def __init__(self, artifact: FateThresholdArtifact) -> None:
        if artifact.threshold_policy != "common":
            raise ValueError("common policy requires a common artifact")
        super().__init__(artifact)
