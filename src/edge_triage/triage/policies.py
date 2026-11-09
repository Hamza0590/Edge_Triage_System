"""Label-free conservative three-way decisions and versioned logical bytes."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from edge_triage.contracts import Contract, Count, PredictionResult, TriageDecision
from edge_triage.runtime.interfaces import FateContext, ThresholdDecision

FALLBACK_DIGEST = hashlib.sha256(b"recall_fate_v1:preserve_on_failure").hexdigest()


class LogicalByteAccounting(Contract):
    version: Literal["logical_raw_tensor_v1"] = "logical_raw_tensor_v1"
    compression_fraction: float = Field(default=0.25, gt=0, le=1)
    metadata_bytes: Count = 0
    # Metadata is part of both original and surviving payloads; protocol overhead=0.
    # No codec or physical network runs, so neither speed nor fidelity is claimed.

    def original(self, tensor_bytes: int) -> int:
        if isinstance(tensor_bytes, bool) or not isinstance(tensor_bytes, int) or tensor_bytes < 0:
            raise ValueError("tensor bytes must be a nonnegative integer")
        return tensor_bytes + self.metadata_bytes

    def output(self, action: str, tensor_bytes: int) -> int:
        original = self.original(tensor_bytes)
        if action == "discard":
            return 0
        if action == "transmit":
            return original
        if action == "compress":
            return math.ceil(tensor_bytes * self.compression_fraction) + self.metadata_bytes
        raise ValueError("unknown fate action")


@dataclass(frozen=True)
class PerTierRecallConstrainedPolicy:
    accounting: LogicalByteAccounting = LogicalByteAccounting()
    binary: bool = False

    def decide(
        self, prediction: PredictionResult, threshold: ThresholdDecision, original_bytes: int
    ) -> TriageDecision:
        # The old interface has no uncertainty evidence, so it cannot authorize discard.
        return self.decide_context(
            FateContext(candidate_id=prediction.candidate_id, prediction=prediction),
            threshold,
            original_bytes,
        )

    def decide_context(
        self, context: FateContext, threshold: ThresholdDecision | None, original_bytes: int
    ) -> TriageDecision:
        start = time.perf_counter_ns()
        prediction, uncertainty = context.prediction, context.uncertainty
        action: Literal["discard", "compress", "transmit"] = "transmit"
        reason = "PRESERVE_MISSING_PREDICTION"
        score = None
        if prediction is not None:
            score = prediction.p_real
            if threshold is not None and threshold.probability_mode == "calibrated":
                score = prediction.calibrated_p_real
        if context.failed:
            reason = "PRESERVE_FAILED_INFERENCE"
        elif threshold is None or threshold.probability_mode is None:
            reason = "PRESERVE_MISSING_FATE_THRESHOLD"
        elif prediction is None:
            pass
        elif prediction.candidate_id != context.candidate_id:
            reason = "PRESERVE_IDENTITY_MISMATCH"
        elif prediction.tier != threshold.trusted_tier:
            reason = "PRESERVE_TIER_MISMATCH"
        elif prediction.checkpoint_id != threshold.checkpoint_id:
            reason = "PRESERVE_CHECKPOINT_MISMATCH"
        elif threshold.probability_mode == "calibrated" and (
            score is None or prediction.calibration_artifact_id != threshold.calibration_artifact_id
        ):
            reason = "PRESERVE_MISSING_CALIBRATION"
        elif (
            threshold.probability_mode == "uncalibrated"
            and prediction.calibrated_p_real is not None
        ):
            reason = "PRESERVE_PROBABILITY_MODE_MISMATCH"
        elif context.conservative_fallback:
            reason = "PRESERVE_FINAL_TIER_AMBIGUITY"
        elif not threshold.discard_supported:
            reason = "PRESERVE_UNVALIDATED_TIER"
        elif uncertainty is None:
            reason = "PRESERVE_MISSING_UNCERTAINTY"
        elif (
            uncertainty.candidate_id != context.candidate_id or uncertainty.tier != prediction.tier
        ):
            reason = "PRESERVE_UNCERTAINTY_IDENTITY_MISMATCH"
        elif uncertainty.ambiguous or uncertainty.unresolved:
            reason = "PRESERVE_UNRESOLVED_UNCERTAINTY"
        elif uncertainty.standard_deviation > threshold.max_uncertainty_for_discard:
            reason = "PRESERVE_HIGH_UNCERTAINTY"
        else:
            assert score is not None
            if score >= threshold.min_p_real_for_transmit:
                reason = "HIGH_P_REAL_TRANSMIT"
            elif (
                threshold.max_p_real_for_discard is not None
                and score <= threshold.max_p_real_for_discard
            ):
                action, reason = "discard", "LOW_P_REAL_TRUSTED_DISCARD"
            elif self.binary:
                reason = "BINARY_SURVIVOR_TRANSMIT"
            else:
                action, reason = "compress", "INTERMEDIATE_P_REAL_LOGICAL_COMPRESS"
        return TriageDecision(
            candidate_id=context.candidate_id,
            action=action,
            threshold_artifact_id=threshold.artifact_id
            if threshold
            else "FATE_FAILURE_FALLBACK_V1",
            threshold_artifact_sha256=threshold.artifact_sha256 if threshold else FALLBACK_DIGEST,
            p_real=score,
            reason_code=reason,
            input_bytes=self.accounting.original(original_bytes),
            output_bytes=self.accounting.output(action, original_bytes),
            duration_ms=(time.perf_counter_ns() - start) / 1e6,
        )


class CommonThresholdThreeWayPolicy(PerTierRecallConstrainedPolicy):
    """Same fate mechanism with a validated common operating point."""


class BinaryDiscardTransmitPolicy(PerTierRecallConstrainedPolicy):
    def __init__(self, accounting: LogicalByteAccounting = LogicalByteAccounting()) -> None:
        super().__init__(accounting, binary=True)


class TransmitAllPolicy(PerTierRecallConstrainedPolicy):
    def decide_context(
        self, context: FateContext, threshold: ThresholdDecision | None, original_bytes: int
    ) -> TriageDecision:
        result = super().decide_context(
            FateContext(
                candidate_id=context.candidate_id, prediction=context.prediction, failed=True
            ),
            threshold,
            original_bytes,
        )
        return TriageDecision.model_validate(
            {**result.model_dump(), "reason_code": "TRANSMIT_ALL_BASELINE"}
        )
