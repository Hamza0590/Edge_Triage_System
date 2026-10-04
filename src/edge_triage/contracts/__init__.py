"""Version 1 shared contracts. Values are metadata, never image/tensor objects.

All durations are milliseconds, memory is bytes, rates are per second,
utilization is percent, and timestamps are timezone-aware UTC.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION: Literal[1] = 1
BOGUS_LABEL = 0
REAL_LABEL = 1
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Percentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0, strict=True)]
PositiveCount = Annotated[int, Field(gt=0, strict=True)]
Identifier = Annotated[str, Field(min_length=1, max_length=256)]
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def p_real_from_logit(logit: float) -> float:
    """Numerically stable sigmoid, including extreme finite logits."""
    if not math.isfinite(logit):
        raise ValueError("logit must be finite")
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1 + exp_value)


class FrozenDict(dict[str, Any]):
    """A JSON-compatible dictionary that cannot be mutated through its public API."""

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("contract mappings are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        # Pickle's default SETITEMS path calls our mutation guard. Reconstruct in
        # the constructor so Windows spawned workers retain immutable config.
        return type(self), (dict(self),)


class FrozenList(list[Any]):
    """Preserve JSON list semantics while preventing ordinary mutation."""

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("contract sequences are immutable")

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = _immutable
    remove = reverse = sort = __iadd__ = __imul__ = _immutable

    def __reduce__(self) -> tuple[Any, tuple[list[Any]]]:
        return type(self), (list(self),)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


class Contract(BaseModel):
    """Frozen, extra-forbidding, finite JSON metadata contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def freeze_and_validate_times(self) -> Self:
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                offset = value.utcoffset()
                if offset is None or offset.total_seconds() != 0:
                    raise ValueError(f"{name} must be timezone-aware UTC")
            if isinstance(value, (dict, list, tuple)):
                object.__setattr__(self, name, _freeze(value))
        return self


class ModelTier(str, Enum):
    TINY = "tiny"
    MEDIUM = "medium"
    LARGE = "large"


class CandidateState(str, Enum):
    QUEUED = "queued"
    ADMITTED = "admitted"
    RUNNING = "running"
    PREDICTED = "predicted"
    ESCALATED = "escalated"
    TRIAGED = "triaged"
    REJECTED = "rejected"
    FAILED = "failed"


class Candidate(Contract):
    """Stable identity and raw-array index; pixels are held separately by executors."""

    run_id: Identifier
    trace_id: Identifier
    candidate_id: Identifier
    dataset_index: Count
    arrival_monotonic_ns: Count
    state: CandidateState = CandidateState.QUEUED
    label: Literal[0, 1] | None = None

    def transition(self, state: CandidateState) -> Candidate:
        allowed = {
            CandidateState.QUEUED: {CandidateState.ADMITTED, CandidateState.REJECTED},
            CandidateState.ADMITTED: {CandidateState.RUNNING, CandidateState.FAILED},
            CandidateState.RUNNING: {CandidateState.PREDICTED, CandidateState.FAILED},
            CandidateState.PREDICTED: {CandidateState.TRIAGED, CandidateState.ESCALATED},
            CandidateState.ESCALATED: {CandidateState.QUEUED, CandidateState.FAILED},
        }
        if state not in allowed.get(self.state, set()):
            raise ValueError(f"illegal candidate transition: {self.state.value} -> {state}")
        return Candidate.model_validate({**self.model_dump(), "state": state})


class PredictionResult(Contract):
    """Raw logit with uncalibrated and optional calibrated probability of Real."""

    candidate_id: Identifier
    tier: ModelTier
    checkpoint_id: Identifier
    logit: float
    p_real: Probability
    calibrated_p_real: Probability | None = None
    calibration_artifact_id: Identifier | None = None
    inference_ms: NonNegative
    preprocessing_ms: NonNegative = 0

    @model_validator(mode="after")
    def check_probability(self) -> Self:
        if not math.isclose(self.p_real, p_real_from_logit(self.logit), abs_tol=1e-6):
            raise ValueError("p_real must equal sigmoid(raw logit)")
        if (self.calibrated_p_real is None) != (self.calibration_artifact_id is None):
            raise ValueError("calibrated probability and artifact must be supplied together")
        return self

    @property
    def p_bogus(self) -> float:
        return 1 - self.p_real


class UncertaintyResult(Contract):
    """MC summaries; policy implementation follows in Module 08."""

    candidate_id: Identifier
    tier: ModelTier
    method: Literal["none", "mc_dropout"]
    passes: PositiveCount
    mean_p_real: Probability
    variance: Annotated[float, Field(ge=0, le=0.25)]
    predictive_entropy: NonNegative
    duration_ms: NonNegative
    standard_deviation: NonNegative = 0
    variation_ratio: Probability = 0
    additional_forward_passes: Count = 0
    ambiguous: bool = False
    unresolved: bool = False
    reason_code: Identifier = "UNCERTAINTY_DISABLED"
    artifact_id: Identifier | None = None
    diagnostic_seed: Count | None = None


class TierJobs(Contract):
    tier: ModelTier
    count: Count


class TierLatency(Contract):
    tier: ModelTier
    mean_ms: NonNegative | None
    samples: Count


class ResourceSnapshot(Contract):
    """Unavailable hardware/network signals are None, never fabricated zeros."""

    timestamp_utc: datetime
    monotonic_ns: Count
    cpu_percent: Percentage | None
    available_ram_bytes: Count | None
    gpu_utilization_percent: Percentage | None = None
    gpu_memory_used_bytes: Count | None = None
    gpu_memory_total_bytes: Count | None = None
    queue_length: Count
    arrival_rate_per_s: NonNegative | None
    active_jobs_by_tier: tuple[TierJobs, ...] = ()
    recent_latency_ms: NonNegative | None = None
    network_available: bool | None = None
    network_bandwidth_bytes_per_s: NonNegative | None = None
    network_latency_ms: NonNegative | None = None
    snapshot_id: Identifier | None = None
    used_ram_bytes: Count | None = None
    process_cpu_percent: NonNegative | None = None
    process_rss_bytes: Count | None = None
    gpu_memory_free_bytes: Count | None = None
    gpu_allocated_bytes: Count | None = None
    gpu_reserved_bytes: Count | None = None
    gpu_temperature_c: float | None = None
    gpu_power_watts: NonNegative | None = None
    gpu_observed_monotonic_ns: Count | None = None
    oldest_wait_ms: NonNegative = 0
    active_jobs_total: Count | None = None
    recent_latency_by_tier: tuple[TierLatency, ...] = ()
    arrival_samples: Count = 0
    latency_samples: Count = 0
    estimator_window_s: NonNegative | None = None
    availability: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_resources(self) -> Self:
        tiers = [item.tier for item in self.active_jobs_by_tier]
        if len(set(tiers)) != len(tiers):
            raise ValueError("duplicate active-job tier")
        latency_tiers = [item.tier for item in self.recent_latency_by_tier]
        if len(set(latency_tiers)) != len(latency_tiers):
            raise ValueError("duplicate latency-history tier")
        if (
            self.gpu_memory_used_bytes is not None
            and self.gpu_memory_total_bytes is not None
            and self.gpu_memory_used_bytes > self.gpu_memory_total_bytes
        ):
            raise ValueError("GPU memory used exceeds total")
        return self


class SchedulingDecision(Contract):
    candidate_id: Identifier
    selected_tier: ModelTier | None
    reason_code: Identifier
    duration_ms: NonNegative = 0
    explanation: dict[str, JsonValue] = Field(default_factory=dict)


class AdmissionDecision(Contract):
    candidate_id: Identifier
    outcome: Literal["admitted", "delayed", "rejected"]
    permitted_concurrency: Count
    reason_code: Identifier
    duration_ms: NonNegative = 0
    explanation: dict[str, JsonValue] = Field(default_factory=dict)
    reservation_id: Identifier | None = None

    @model_validator(mode="after")
    def check_admission(self) -> Self:
        if self.outcome == "admitted" and self.permitted_concurrency == 0:
            raise ValueError("admitted work requires positive permitted concurrency")
        return self


class EscalationDecision(Contract):
    candidate_id: Identifier
    from_tier: ModelTier
    to_tier: ModelTier | None = None
    escalate: bool
    reason_code: Identifier
    duration_ms: NonNegative = 0
    conservative_fallback: bool = False
    visited_tiers: tuple[ModelTier, ...] = ()

    @model_validator(mode="after")
    def check_escalation(self) -> Self:
        order = list(ModelTier)
        if self.escalate:
            if self.to_tier is None or order.index(self.to_tier) <= order.index(self.from_tier):
                raise ValueError("escalation must target a larger tier")
        elif self.to_tier is not None:
            raise ValueError("non-escalating decision cannot specify a target")
        return self


class TriageDecision(Contract):
    candidate_id: Identifier
    action: Literal["discard", "compress", "transmit"]
    threshold_artifact_id: Identifier
    threshold_artifact_sha256: SHA256
    p_real: Probability
    reason_code: Identifier
    input_bytes: Count
    output_bytes: Count
    duration_ms: NonNegative = 0

    @model_validator(mode="after")
    def check_bytes(self) -> Self:
        if self.action == "discard" and self.output_bytes != 0:
            raise ValueError("discard has zero output bytes")
        if self.action == "compress" and self.output_bytes > self.input_bytes:
            raise ValueError("logical compression cannot expand data")
        if self.action == "transmit" and self.output_bytes != self.input_bytes:
            raise ValueError("transmit preserves input byte count")
        return self


class CandidateEvent(Contract):
    """Schema v1 append-only event; run-level correlation IDs may be None."""

    schema_version: Literal[1] = SCHEMA_VERSION
    timestamp_utc: datetime
    monotonic_ns: Count
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    run_id: Identifier
    trace_id: Identifier | None
    candidate_id: Identifier | None
    module: Identifier
    event_type: Annotated[
        str,
        Field(
            pattern=(
                r"^(run|data|model|monitor|scheduler|inference|uncertainty|triage|experiment|error)"
                r"\.[a-z][a-z0-9_.]*$"
            )
        ),
    ]
    config_hash: SHA256
    message: Annotated[str, Field(max_length=1024)]
    reason_code: Identifier
    duration_ms: NonNegative | None
    payload: dict[str, JsonValue]
    error_type: Annotated[str, Field(max_length=256)] | None
    error_message: Annotated[str, Field(max_length=1024)] | None


class ArtifactHash(Contract):
    """Content identity; missing artifacts are absent rather than fake hashes."""

    artifact_id: Identifier
    path: str
    algorithm: Literal["md5", "sha256"]
    digest: str

    @model_validator(mode="after")
    def check_digest(self) -> Self:
        length = 32 if self.algorithm == "md5" else 64
        if len(self.digest) != length or any(c not in "0123456789abcdef" for c in self.digest):
            raise ValueError("invalid content digest")
        return self


class RunManifest(Contract):
    """Run lifecycle: running -> completed/failed; final manifests are sealed by manager."""

    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: Identifier
    purpose: Identifier
    started_utc: datetime
    ended_utc: datetime | None = None
    status: Literal["running", "completed", "failed"] = "running"
    config_hash: SHA256
    dataset_hashes: tuple[ArtifactHash, ...]
    split_hash: ArtifactHash | None = None
    preprocessing_hash: ArtifactHash | None = None
    checkpoint_hashes: tuple[ArtifactHash, ...] = ()
    threshold_hashes: tuple[ArtifactHash, ...] = ()
    arrival_trace_hash: ArtifactHash | None = None
    git_commit: str | None
    git_dirty: bool | None
    environment: dict[str, JsonValue]
    seeds: dict[str, Count | None]
    provenance_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def check_lifecycle(self) -> Self:
        if (self.status == "running") != (self.ended_utc is None):
            raise ValueError("only running manifests have no end time")
        if self.ended_utc is not None and self.ended_utc < self.started_utc:
            raise ValueError("end time precedes start time")
        return self


class ModelProfile(Contract):
    """Hardware-specific measurements; no assumed cross-device portability."""

    profile_id: Identifier
    tier: ModelTier
    checkpoint_id: Identifier
    device: Identifier
    batch_size: PositiveCount
    concurrency: PositiveCount
    input_shape: tuple[PositiveCount, PositiveCount, PositiveCount]
    latency_p50_ms: NonNegative
    latency_p95_ms: NonNegative
    peak_memory_bytes: Count
    samples: PositiveCount
    measured_utc: datetime

    @model_validator(mode="after")
    def check_quantiles(self) -> Self:
        if self.latency_p95_ms < self.latency_p50_ms:
            raise ValueError("p95 latency cannot be below p50")
        return self
