"""Deterministic non-scientific policies for interface tests and future smoke runs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from edge_triage.contracts import (
    AdmissionDecision,
    Candidate,
    EscalationDecision,
    ModelTier,
    PredictionResult,
    ResourceSnapshot,
    SchedulingDecision,
    TriageDecision,
    UncertaintyResult,
)
from edge_triage.runtime.interfaces import ThresholdDecision

SMOKE_PURPOSE = "NON_SCIENTIFIC_SMOKE_TEST"


@dataclass(frozen=True)
class StaticTierPolicy:
    tier: ModelTier

    def select(self, candidate: Candidate, resources: ResourceSnapshot) -> SchedulingDecision:
        return SchedulingDecision(
            candidate_id=candidate.candidate_id, selected_tier=self.tier, reason_code="STATIC_TIER"
        )


@dataclass(frozen=True)
class SequentialAdmissionPolicy:
    max_active: int = 1

    def __post_init__(self) -> None:
        if self.max_active != 1:
            raise ValueError("sequential admission requires max_active=1")

    def decide(self, candidate: Candidate, resources: ResourceSnapshot) -> AdmissionDecision:
        available = sum(item.count for item in resources.active_jobs_by_tier) < self.max_active
        return AdmissionDecision(
            candidate_id=candidate.candidate_id,
            outcome="admitted" if available else "delayed",
            permitted_concurrency=self.max_active,
            reason_code="SEQUENTIAL_CAPACITY" if available else "WAIT_ACTIVE_JOB",
        )


class NoUncertaintyPolicy:
    def evaluate(self, prediction: PredictionResult) -> UncertaintyResult:
        p = (
            prediction.p_real
            if prediction.calibrated_p_real is None
            else prediction.calibrated_p_real
        )
        entropy = -sum(value * math.log(value) for value in (p, 1 - p) if value > 0)
        return UncertaintyResult(
            candidate_id=prediction.candidate_id,
            tier=prediction.tier,
            method="none",
            passes=1,
            mean_p_real=p,
            variance=0,
            predictive_entropy=entropy,
            duration_ms=0,
        )


class NoEscalationPolicy:
    def decide(
        self, prediction: PredictionResult, uncertainty: UncertaintyResult
    ) -> EscalationDecision:
        return EscalationDecision(
            candidate_id=prediction.candidate_id,
            from_tier=prediction.tier,
            escalate=False,
            reason_code="ESCALATION_DISABLED",
        )


@dataclass(frozen=True)
class CommonThresholdPolicy:
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be a finite probability")

    def definition(self) -> bytes:
        return f"{SMOKE_PURPOSE}:common-threshold-v1:{self.threshold:.17g}".encode()

    def resolve(self, tier: ModelTier) -> ThresholdDecision:
        return ThresholdDecision(
            p_real_threshold=self.threshold,
            artifact_id="NON_SCIENTIFIC_COMMON_THRESHOLD_V1",
            artifact_sha256=hashlib.sha256(self.definition()).hexdigest(),
        )


class ClassificationOnlyFatePolicy:
    """Record a classification but transmit all bytes; never claim recall calibration."""

    def decide(
        self, prediction: PredictionResult, threshold: ThresholdDecision, original_bytes: int
    ) -> TriageDecision:
        predicted_real = prediction.p_real >= threshold.p_real_threshold
        return TriageDecision(
            candidate_id=prediction.candidate_id,
            action="transmit",
            threshold_artifact_id=threshold.artifact_id,
            threshold_artifact_sha256=threshold.artifact_sha256,
            p_real=prediction.p_real,
            reason_code="CLASSIFIED_REAL" if predicted_real else "CLASSIFIED_BOGUS",
            input_bytes=original_bytes,
            output_bytes=original_bytes,
        )
