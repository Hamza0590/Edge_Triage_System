"""Label-free selective UQ and strictly increasing portfolio cascades."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from edge_triage.calibration.temperature import CalibratedModelAdapter
from edge_triage.contracts import EscalationDecision, ModelTier, PredictionResult, UncertaintyResult
from edge_triage.runtime.reference import NoUncertaintyPolicy
from edge_triage.uncertainty.artifacts import UncertaintyArtifact
from edge_triage.uncertainty.sampling import sample_probabilities, summarize


def trusted_probability(prediction: PredictionResult) -> float:
    return (
        prediction.p_real if prediction.calibrated_p_real is None else prediction.calibrated_p_real
    )


def diagnostic_seed(seed: int | None, candidate_id: str, tier: ModelTier) -> int | None:
    if seed is None:
        return None
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{candidate_id}:{tier.value}".encode()).digest()[:8], "big"
    ) % (2**63)


class AmbiguousOnlyMCDropoutPolicy:
    always = False

    def __init__(
        self,
        models: dict[ModelTier, Any],
        artifact: UncertaintyArtifact,
        seed: int | None = None,
    ) -> None:
        self.models, self.artifact, self.seed = models, artifact, seed
        self.tiers = {t.tier: t for t in artifact.tiers}
        for tier, adapter in models.items():
            spec = self.tiers[tier]
            if not isinstance(adapter, CalibratedModelAdapter):
                raise ValueError("UQ requires an explicitly calibrated/fallback adapter")
            if (
                adapter.base.checkpoint_id != spec.checkpoint_id
                or adapter.artifact.artifact_id != spec.calibration_artifact_id
                or adapter.mode != spec.probability_mode
                or adapter.artifact.temperature != spec.temperature
            ):
                raise ValueError("UQ probability/checkpoint/calibration mismatch")

    def evaluate(self, prediction: PredictionResult) -> UncertaintyResult:
        raise ValueError("MC policy requires label-free image input")

    def evaluate_image(self, prediction: PredictionResult, image: Any) -> UncertaintyResult:
        start = time.perf_counter_ns()
        spec = self.tiers[prediction.tier]
        expected_calibration = (
            spec.calibration_artifact_id if spec.probability_mode == "calibrated" else None
        )
        if (
            prediction.checkpoint_id != spec.checkpoint_id
            or prediction.calibration_artifact_id != expected_calibration
        ):
            raise ValueError("prediction does not match frozen UQ artifact")
        p = trusted_probability(prediction)
        ambiguous = spec.lower <= p <= spec.upper
        if not ambiguous and not self.always:
            return (
                NoUncertaintyPolicy()
                .evaluate(prediction)
                .model_copy(
                    update={
                        "artifact_id": self.artifact.artifact_id,
                        "reason_code": "OUTSIDE_AMBIGUOUS_ZONE",
                        "duration_ms": (time.perf_counter_ns() - start) / 1e6,
                    }
                )
            )
        seed = diagnostic_seed(self.seed, prediction.candidate_id, prediction.tier)
        adapter = self.models[prediction.tier]
        values, elapsed = sample_probabilities(
            adapter.base.model,
            image,
            self.artifact.passes,
            temperature=spec.temperature if spec.probability_mode == "calibrated" else 1,
            seed=seed,
        )
        result = summarize(prediction, values, elapsed)
        disagreement = (result.mean_p_real >= 0.5) != (p >= 0.5)
        high = result.standard_deviation >= spec.std_threshold
        # Ambiguity alone invokes sampling; high dispersion or disagreement routes onward.
        unresolved = high or disagreement
        return result.model_copy(
            update={
                "ambiguous": ambiguous,
                "unresolved": unresolved,
                "reason_code": "MC_DISAGREEMENT"
                if disagreement
                else "MC_HIGH_STD"
                if high
                else "MC_RESOLVED",
                "artifact_id": self.artifact.artifact_id,
                "diagnostic_seed": seed,
                "duration_ms": (time.perf_counter_ns() - start) / 1e6,
            }
        )


class AlwaysMCDropoutPolicy(AmbiguousOnlyMCDropoutPolicy):
    """Explicit expensive ablation; not the default policy."""

    always = True


class CascadeEscalationPolicy:
    def __init__(self, portfolio: tuple[ModelTier, ...], maximum: int = 2) -> None:
        if not portfolio or len(set(portfolio)) != len(portfolio) or maximum < 0:
            raise ValueError("unique nonempty portfolio and nonnegative maximum required")
        self.portfolio = tuple(t for t in ModelTier if t in portfolio)
        self.maximum = maximum

    def decide(
        self, prediction: PredictionResult, uncertainty: UncertaintyResult
    ) -> EscalationDecision:
        return self.decide_path(prediction, uncertainty, (prediction.tier,))

    def decide_path(
        self,
        prediction: PredictionResult,
        uncertainty: UncertaintyResult,
        visited: tuple[ModelTier, ...],
    ) -> EscalationDecision:
        start = time.perf_counter_ns()
        order = list(ModelTier)
        if (
            not visited
            or visited[-1] != prediction.tier
            or any(t not in self.portfolio for t in visited)
            or any(order.index(a) >= order.index(b) for a, b in zip(visited, visited[1:]))
            or uncertainty.candidate_id != prediction.candidate_id
            or uncertainty.tier != prediction.tier
        ):
            raise ValueError("invalid visited path or uncertainty identity")
        index = self.portfolio.index(prediction.tier)
        target = None
        reason = "UQ_RESOLVED"
        fallback = False
        if uncertainty.unresolved:
            if len(visited) - 1 >= self.maximum:
                reason, fallback = "MAX_ESCALATIONS_CONSERVATIVE", True
            elif index == len(self.portfolio) - 1:
                reason, fallback = "FINAL_TIER_CONSERVATIVE", True
            else:
                target, reason = self.portfolio[index + 1], uncertainty.reason_code
        return EscalationDecision(
            candidate_id=prediction.candidate_id,
            from_tier=prediction.tier,
            to_tier=target,
            escalate=target is not None,
            reason_code=reason,
            conservative_fallback=fallback,
            visited_tiers=visited,
            duration_ms=(time.perf_counter_ns() - start) / 1e6,
        )
