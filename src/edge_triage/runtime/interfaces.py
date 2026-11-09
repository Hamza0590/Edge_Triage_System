"""Typed pipeline boundaries; tensor-bearing inputs are never event payloads."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import Field

from edge_triage.contracts import (
    SHA256,
    AdmissionDecision,
    Candidate,
    CandidateEvent,
    Contract,
    EscalationDecision,
    ModelTier,
    PredictionResult,
    Probability,
    ResourceSnapshot,
    SchedulingDecision,
    TriageDecision,
    UncertaintyResult,
)

if TYPE_CHECKING:
    import torch

    from edge_triage.runtime.executors import InferenceJob, JobHandle


@dataclass(frozen=True)
class CandidateInput:
    """In-process work item. Offline truth stays separate from scheduler metadata."""

    candidate: Candidate
    image: torch.Tensor
    label: int
    original_bytes: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1) or self.original_bytes < 0:
            raise ValueError("invalid offline label or original byte count")
        if self.candidate.label is not None:
            raise ValueError("runtime Candidate metadata must not contain ground truth")


class ThresholdDecision(Contract):
    """Reference classification threshold; calibration is a later module."""

    p_real_threshold: Probability
    artifact_id: str
    artifact_sha256: SHA256
    max_p_real_for_discard: float | None = Field(default=None, ge=0, lt=0.5)
    min_p_real_for_transmit: float = Field(default=0.5, ge=0.5, le=1)
    max_uncertainty_for_discard: float = Field(default=0, ge=0, le=0.5)
    checkpoint_id: str | None = None
    trusted_tier: ModelTier | None = None
    discard_supported: bool = False
    probability_mode: Literal["calibrated", "uncalibrated"] | None = None
    calibration_artifact_id: str | None = None


class FateContext(Contract):
    """Final runtime evidence, deliberately without ground-truth labels."""

    candidate_id: str
    prediction: PredictionResult | None = None
    uncertainty: UncertaintyResult | None = None
    conservative_fallback: bool = False
    failed: bool = False


@runtime_checkable
class ContextFatePolicy(Protocol):
    def decide_context(
        self, context: FateContext, threshold: ThresholdDecision | None, original_bytes: int
    ) -> TriageDecision: ...


class CandidateSource(Protocol):
    def __iter__(self) -> Iterator[CandidateInput]: ...


class CandidateQueue(Protocol):
    def put(self, item: CandidateInput) -> None: ...
    def get(self) -> CandidateInput: ...
    def __len__(self) -> int: ...


class TierSelectionPolicy(Protocol):
    def select(self, candidate: Candidate, resources: ResourceSnapshot) -> SchedulingDecision: ...


class AdmissionPolicy(Protocol):
    def decide(self, candidate: Candidate, resources: ResourceSnapshot) -> AdmissionDecision: ...


@runtime_checkable
class TierAdmissionPolicy(Protocol):
    """Optional lease-aware extension; legacy two-argument policies remain substitutable."""

    def admit(
        self, candidate: Candidate, selected_tier: ModelTier, resources: ResourceSnapshot
    ) -> AdmissionDecision: ...
    def release(self, reservation_id: str, *, outcome: str = "completed") -> None: ...


class ModelAdapter(Protocol):
    def predict(self, item: CandidateInput, tier: ModelTier) -> PredictionResult: ...


class InferenceExecutor(Protocol):
    def execute(
        self, model: ModelAdapter, item: CandidateInput, tier: ModelTier
    ) -> PredictionResult: ...
    def close(self) -> None: ...


class AsyncInferenceExecutor(Protocol):
    """Module09 job lifecycle; legacy blocking pipeline protocol remains compatible."""

    async def submit(self, job: InferenceJob) -> JobHandle: ...
    async def cancel(self, job_id: str) -> bool: ...
    async def drain(self) -> None: ...
    async def shutdown(self, *, cancel_pending: bool = False) -> None: ...
    def notify_resources(self) -> None: ...


class UncertaintyPolicy(Protocol):
    def evaluate(self, prediction: PredictionResult) -> UncertaintyResult: ...


@runtime_checkable
class ImageUncertaintyPolicy(Protocol):
    """Only pixels and prediction enter UQ; no CandidateInput or offline truth."""

    def evaluate_image(
        self, prediction: PredictionResult, image: torch.Tensor
    ) -> UncertaintyResult: ...


@runtime_checkable
class PathEscalationPolicy(Protocol):
    def decide_path(
        self,
        prediction: PredictionResult,
        uncertainty: UncertaintyResult,
        visited: tuple[ModelTier, ...],
    ) -> EscalationDecision: ...


class EscalationPolicy(Protocol):
    def decide(
        self, prediction: PredictionResult, uncertainty: UncertaintyResult
    ) -> EscalationDecision: ...


class ThresholdPolicy(Protocol):
    def resolve(self, tier: ModelTier) -> ThresholdDecision: ...


class FatePolicy(Protocol):
    def decide(
        self, prediction: PredictionResult, threshold: ThresholdDecision, original_bytes: int
    ) -> TriageDecision: ...


class EventSink(Protocol):
    def emit(self, event: CandidateEvent) -> None: ...
