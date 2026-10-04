"""Sequential reference orchestration composed from independent typed policies."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import psutil

from edge_triage.config import AppConfig
from edge_triage.contracts import (
    AdmissionDecision,
    Candidate,
    Contract,
    ModelTier,
    PredictionResult,
    ResourceSnapshot,
    SchedulingDecision,
    TriageDecision,
    UncertaintyResult,
    utc_now,
)
from edge_triage.event_logging import EventLogger
from edge_triage.monitoring.estimators import RuntimeCounters
from edge_triage.runtime.components import LoggerEventSink
from edge_triage.runtime.executors import FixedConcurrencyExecutor, InferenceJob
from edge_triage.runtime.interfaces import (
    AdmissionPolicy,
    CandidateInput,
    CandidateQueue,
    CandidateSource,
    EscalationPolicy,
    FatePolicy,
    ImageUncertaintyPolicy,
    InferenceExecutor,
    ModelAdapter,
    PathEscalationPolicy,
    ThresholdPolicy,
    TierAdmissionPolicy,
    TierSelectionPolicy,
    UncertaintyPolicy,
)
from edge_triage.runtime.records import CandidateRecord, ParquetRecordWriter
from edge_triage.runtime.state import PipelineState, StateMachine
from edge_triage.uncertainty.sampling import SamplingError


class AdmissionTimeoutError(RuntimeError):
    pass


class AdmissionRejectedError(RuntimeError):
    pass


class NoFeasibleTierError(RuntimeError):
    pass


@dataclass
class PipelineComponents:
    queue: CandidateQueue
    selector: TierSelectionPolicy
    admission: AdmissionPolicy
    executor: InferenceExecutor
    models: dict[ModelTier, ModelAdapter]
    uncertainty: UncertaintyPolicy
    escalation: EscalationPolicy
    thresholds: ThresholdPolicy
    fate: FatePolicy


def minimal_snapshot(queue_length: int) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp_utc=utc_now(),
        monotonic_ns=time.monotonic_ns(),
        cpu_percent=psutil.cpu_percent(),
        available_ram_bytes=psutil.virtual_memory().available,
        queue_length=queue_length,
        arrival_rate_per_s=0,
    )


@dataclass
class QueueEntry:
    machine: StateMachine
    arrival_utc: str
    queue_ns: int


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        components: PipelineComponents,
        logger: EventLogger,
        writer: ParquetRecordWriter,
        *,
        snapshot: Callable[[int], ResourceSnapshot] = minimal_snapshot,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config, self.parts, self.logger, self.writer = config, components, logger, writer
        self.snapshot, self.sleep = snapshot, sleep

    def _event(
        self,
        candidate: Candidate,
        family: str,
        reason: str,
        payload: Contract | dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if isinstance(payload, ResourceSnapshot) and payload.snapshot_id is not None:
            payload = {"snapshot_id": payload.snapshot_id, "monotonic_ns": payload.monotonic_ns}
        if isinstance(payload, CandidateRecord):
            # Structured bounded chains avoid the logger's 1024-character string limit.
            payload = payload.model_dump(mode="json")
            for name in ("prediction", "uncertainty", "escalation", "execution"):
                payload[f"{name}_chain"] = json.loads(payload.pop(f"{name}_chain_json"))
            payload["resource_snapshot_refs"] = [
                {
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "monotonic_ns": snapshot["monotonic_ns"],
                }
                for snapshot in json.loads(payload.pop("resource_snapshot_json"))
            ]
        self.logger.bind(candidate.trace_id, candidate.candidate_id).emit(
            module="runtime.pipeline",
            event_type=family,
            reason_code=reason,
            message=reason,
            payload=payload,
            exception=error,
            level="ERROR" if error else "INFO",
        )

    def run(self, source: CandidateSource) -> tuple[int, int]:
        entries: dict[str, QueueEntry] = {}
        completed = failed = 0
        try:
            for item in source:
                candidate = item.candidate
                if candidate.run_id != self.logger.run_id or candidate.candidate_id in entries:
                    raise ValueError("source run mismatch or duplicate candidate ID")
                queued = time.monotonic_ns()
                if candidate.arrival_monotonic_ns == 0:
                    candidate = Candidate.model_validate(
                        {**candidate.model_dump(), "arrival_monotonic_ns": queued}
                    )
                    item = CandidateInput(candidate, item.image, item.label, item.original_bytes)
                if candidate.arrival_monotonic_ns > queued:
                    raise ValueError("arrival timestamp cannot be in the future")
                machine = StateMachine(
                    candidate, self.logger.config_hash, LoggerEventSink(self.logger)
                )
                entries[candidate.candidate_id] = QueueEntry(machine, utc_now().isoformat(), queued)
                self._event(candidate, "data.arrived", "CANDIDATE_ARRIVED")
                self.parts.queue.put(item)
                machine.transition(PipelineState.QUEUED)
                self._event(candidate, "scheduler.queued", "FIFO_ENQUEUED")
            while len(self.parts.queue):
                item = self.parts.queue.get()
                record = self._process(item, entries[item.candidate.candidate_id])
                self.writer.append(record)
                self._event(item.candidate, "inference.candidate_final", "CANDIDATE_FINAL", record)
                completed += int(record.status == "completed")
                failed += int(record.status == "failed")
            return completed, failed
        finally:
            try:
                self.parts.executor.close()
            finally:
                self.writer.close()

    def _process(self, item: CandidateInput, entry: QueueEntry) -> CandidateRecord:
        return asyncio.run(self.process_async(item, entry))

    async def process_async(
        self,
        item: CandidateInput,
        entry: QueueEntry,
        executor: FixedConcurrencyExecutor | None = None,
        counters: RuntimeCounters | None = None,
    ) -> CandidateRecord:
        candidate, machine = item.candidate, entry.machine
        selected: list[ModelTier] = []
        snapshots: list[ResourceSnapshot] = []
        predictions: list[PredictionResult] = []
        uq_results: list[UncertaintyResult] = []
        escalation_results: list[dict[str, Any]] = []
        uq_ms = escalation_ms = admission_ms = executor_ms = 0.0
        executions: list[dict[str, Any]] = []
        failed_uq_attempts = failed_uq_completed = 0
        conservative = False
        uncertainty: UncertaintyResult | None = None
        fate: TriageDecision | None = None
        scheduled: int | None = None
        started: int | None = None
        ended: int | None = None
        inference_ms = 0.0
        active_jobs = concurrency = 0
        status: Literal["completed", "failed"] = "completed"
        error_code: str | None = None
        next_tier: ModelTier | None = None
        reservation: str | None = None
        queue_length = len(self.parts.queue) + 1

        def verify_identity(identity: str) -> None:
            if identity != candidate.candidate_id:
                raise ValueError("policy/executor changed candidate identity")

        try:
            while True:
                resources = self.snapshot(len(self.parts.queue) + 1)
                snapshots.append(resources)
                self._event(candidate, "monitor.snapshot", "REFERENCE_SNAPSHOT", resources)
                decision = (
                    self.parts.selector.select(
                        candidate,
                        resources.model_copy(
                            update={"active_jobs_by_tier": (), "active_jobs_total": 0}
                        )
                        if executor is not None
                        else resources,
                    )
                    if next_tier is None
                    else SchedulingDecision(
                        candidate_id=candidate.candidate_id,
                        selected_tier=next_tier,
                        reason_code="ESCALATION_TARGET",
                    )
                )
                verify_identity(decision.candidate_id)
                tier = decision.selected_tier
                self._event(candidate, "scheduler.selected", decision.reason_code, decision)
                if tier is None:
                    raise NoFeasibleTierError("scheduler found no feasible tier")
                if tier not in self.parts.models:
                    raise ValueError("selected tier is outside the configured portfolio")
                if tier in selected or len(selected) > self.config.escalation.max_escalations:
                    raise ValueError("cycle or maximum escalation count exceeded")
                if selected and list(ModelTier).index(tier) <= list(ModelTier).index(selected[-1]):
                    raise ValueError("escalation must increase tier")
                selected.append(tier)
                machine.transition(PipelineState.SCHEDULED)
                scheduled = scheduled or time.monotonic_ns()
                async_result = None
                if executor is not None:
                    launched = False

                    def on_launch(admitted: AdmissionDecision | None, active: int) -> None:
                        nonlocal active_jobs, concurrency, launched
                        if admitted is not None:
                            self._event(
                                candidate, "scheduler.admission", admitted.reason_code, admitted
                            )
                        machine.transition(PipelineState.ADMITTED)
                        machine.transition(PipelineState.RUNNING)
                        active_jobs = max(active_jobs, active)
                        concurrency = (
                            admitted.permitted_concurrency if admitted else executor.concurrency
                        )
                        if counters is not None:
                            counters.started(tier)
                        launched = True

                    handle = await executor.submit(
                        InferenceJob(
                            f"{candidate.candidate_id}:{len(selected)}",
                            item,
                            tier,
                            self.parts.models[tier],
                            self.parts.uncertainty,
                            on_launch,
                        )
                    )
                    async_result = await handle.result()
                    if launched and counters is not None:
                        counters.finished(tier, async_result.inference_ms)
                    # Keep compact identity/cost evidence; predictions/UQ have their own chains.
                    execution = async_result.model_dump(
                        mode="json", exclude={"prediction", "uncertainty", "admission"}
                    )
                    executions.append(execution)
                    self._event(candidate, "inference.execution", "TIER_JOB_FINISHED", execution)
                    admission_ms += async_result.admission_ms
                    executor_ms += async_result.executor_ms
                    uq_ms += async_result.uq_ms
                    inference_ms += async_result.inference_ms
                    started = started or async_result.started_ns
                    ended = async_result.finished_ns if async_result.started_ns else ended
                    if async_result.status != "completed":
                        if async_result.prediction is not None:
                            predictions.append(async_result.prediction)
                        failed_uq_attempts += async_result.additional_forward_passes
                        failed_uq_completed += async_result.failed_uq_completed
                        raise RuntimeError(
                            f"tier job {async_result.status}: {async_result.error_type}"
                        )
                    assert async_result.prediction is not None
                    prediction = async_result.prediction
                else:
                    deadline = time.monotonic() + self.config.pipeline.admission_timeout_s
                    while True:
                        admission = (
                            self.parts.admission.admit(candidate, tier, resources)
                            if isinstance(self.parts.admission, TierAdmissionPolicy)
                            else self.parts.admission.decide(candidate, resources)
                        )
                        reservation = admission.reservation_id
                        verify_identity(admission.candidate_id)
                        self._event(
                            candidate, "scheduler.admission", admission.reason_code, admission
                        )
                        concurrency = admission.permitted_concurrency
                        if admission.outcome == "admitted":
                            break
                        if admission.outcome == "rejected":
                            raise AdmissionRejectedError("candidate rejected by admission policy")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise AdmissionTimeoutError("admission deadline exceeded")
                        self.sleep(min(self.config.pipeline.admission_retry_s, remaining))
                        resources = self.snapshot(len(self.parts.queue) + 1)
                        self._event(
                            candidate, "monitor.snapshot", "ADMISSION_RETRY_SNAPSHOT", resources
                        )
                    machine.transition(PipelineState.ADMITTED)
                    machine.transition(PipelineState.RUNNING)
                    begin = time.monotonic_ns()
                    started = started or begin
                    active_jobs = 1
                    try:
                        prediction = self.parts.executor.execute(
                            self.parts.models[tier], item, tier
                        )
                    except BaseException:
                        if reservation and isinstance(self.parts.admission, TierAdmissionPolicy):
                            self.parts.admission.release(reservation, outcome="failed")
                            reservation = None
                        raise
                    finally:
                        ended = time.monotonic_ns()
                        inference_ms += (ended - begin) / 1e6
                verify_identity(prediction.candidate_id)
                if prediction.tier != tier:
                    raise ValueError("executor changed selected tier")
                predictions.append(prediction)
                machine.transition(PipelineState.PREDICTED)
                self._event(candidate, "inference.predicted", "MODEL_PREDICTION", prediction)
                machine.transition(PipelineState.UNCERTAINTY)
                if async_result is not None:
                    assert async_result.uncertainty is not None
                    uncertainty = async_result.uncertainty
                else:
                    uq_start = time.perf_counter_ns()
                    try:
                        uncertainty = (
                            self.parts.uncertainty.evaluate_image(prediction, item.image)
                            if isinstance(self.parts.uncertainty, ImageUncertaintyPolicy)
                            else self.parts.uncertainty.evaluate(prediction)
                        )
                    except SamplingError as error:
                        failed_uq_attempts += error.attempted
                        failed_uq_completed += error.completed
                        self._event(
                            candidate,
                            "uncertainty.failed",
                            "MC_SAMPLING_FAILED",
                            {
                                "tier": tier.value,
                                "attempted_forward_passes": error.attempted,
                                "completed_forward_passes": error.completed,
                                "duration_ms": error.duration_ms,
                            },
                            error=error,
                        )
                        raise
                    finally:
                        uq_ms += (time.perf_counter_ns() - uq_start) / 1e6
                verify_identity(uncertainty.candidate_id)
                if uncertainty.tier != tier:
                    raise ValueError("UQ changed selected tier")
                uq_results.append(uncertainty)
                if reservation and isinstance(self.parts.admission, TierAdmissionPolicy):
                    self.parts.admission.release(reservation, outcome="completed")
                    reservation = None
                self._event(candidate, "uncertainty.evaluated", "UNCERTAINTY_POLICY", uncertainty)
                escalation_start = time.perf_counter_ns()
                try:
                    escalation = (
                        self.parts.escalation.decide_path(prediction, uncertainty, tuple(selected))
                        if isinstance(self.parts.escalation, PathEscalationPolicy)
                        else self.parts.escalation.decide(prediction, uncertainty)
                    )
                finally:
                    escalation_ms += (time.perf_counter_ns() - escalation_start) / 1e6
                verify_identity(escalation.candidate_id)
                if escalation.from_tier != tier:
                    raise ValueError("escalation changed source tier")
                escalation_results.append(escalation.model_dump(mode="json"))
                conservative = escalation.conservative_fallback
                self._event(candidate, "uncertainty.escalation", escalation.reason_code, escalation)
                if escalation.escalate:
                    if escalation.from_tier != tier or escalation.to_tier not in self.parts.models:
                        raise ValueError("invalid escalation within configured portfolio")
                    next_tier = escalation.to_tier
                    machine.transition(PipelineState.ESCALATED)
                    continue
                threshold = self.parts.thresholds.resolve(tier)
                self._event(candidate, "triage.threshold", "THRESHOLD_ARTIFACT", threshold)
                fate = self.parts.fate.decide(prediction, threshold, item.original_bytes)
                verify_identity(fate.candidate_id)
                machine.transition(PipelineState.TRIAGED)
                self._event(candidate, "triage.decided", fate.reason_code, fate)
                machine.transition(PipelineState.COMPLETED)
                break
        except Exception as error:
            status, error_code = "failed", type(error).__name__
            if reservation and isinstance(self.parts.admission, TierAdmissionPolicy):
                self.parts.admission.release(reservation, outcome="failed")
                reservation = None
            self._event(candidate, "error.candidate_failed", "CANDIDATE_FAILED", error=error)
            machine.transition(PipelineState.FAILED)
        finally:
            if reservation and isinstance(self.parts.admission, TierAdmissionPolicy):
                self.parts.admission.release(reservation, outcome="cancelled")
        final = time.monotonic_ns()
        first, last = (predictions[0], predictions[-1]) if predictions else (None, None)
        return CandidateRecord.model_validate(
            {
                "run_id": candidate.run_id,
                "trace_id": candidate.trace_id,
                "candidate_id": candidate.candidate_id,
                "label": item.label,
                "arrival_utc": entry.arrival_utc,
                "final_utc": utc_now().isoformat(),
                "arrival_ns": candidate.arrival_monotonic_ns,
                "queue_ns": entry.queue_ns,
                "scheduled_ns": scheduled,
                "start_ns": started,
                "end_ns": ended,
                "final_ns": final,
                "selected_tiers": ",".join(tier.value for tier in selected),
                "initial_logit": first.logit if first else None,
                "final_logit": last.logit if last else None,
                "initial_p_real": first.p_real if first else None,
                "final_p_real": last.p_real if last else None,
                "uncertainty_variance": uncertainty.variance if uncertainty else None,
                "uncertainty_method": uncertainty.method if uncertainty else None,
                "escalation_path": "->".join(tier.value for tier in selected),
                "prediction_chain_json": json.dumps(
                    [p.model_dump(mode="json") for p in predictions]
                ),
                "uncertainty_chain_json": json.dumps(
                    [u.model_dump(mode="json") for u in uq_results]
                ),
                "escalation_chain_json": json.dumps(escalation_results),
                "final_trusted_tier": last.tier.value if last else None,
                "final_trusted_p_real": (
                    last.p_real if last.calibrated_p_real is None else last.calibrated_p_real
                )
                if last
                else None,
                "conservative_fallback": int(conservative),
                "uq_ms": uq_ms,
                "admission_ms": admission_ms,
                "executor_ms": executor_ms,
                "execution_chain_json": json.dumps(executions),
                "escalation_decision_ms": escalation_ms,
                "additional_forward_passes": (
                    sum(u.additional_forward_passes for u in uq_results) + failed_uq_attempts
                ),
                "failed_uq_attempts": failed_uq_attempts,
                "failed_uq_completed": failed_uq_completed,
                "resource_snapshot_json": json.dumps(
                    [
                        {"snapshot_id": snap.snapshot_id, "monotonic_ns": snap.monotonic_ns}
                        if snap.snapshot_id is not None
                        else snap.model_dump(mode="json")
                        for snap in snapshots
                    ]
                ),
                "queue_length": queue_length,
                "active_jobs": active_jobs,
                "concurrency": concurrency,
                "fate": fate.action if fate else None,
                "threshold_artifact_id": fate.threshold_artifact_id if fate else None,
                "threshold_artifact_sha256": fate.threshold_artifact_sha256 if fate else None,
                "input_bytes": item.original_bytes,
                "output_bytes": fate.output_bytes if fate else 0,
                "queue_ms": ((started or final) - entry.queue_ns) / 1e6,
                "inference_ms": inference_ms,
                "end_to_end_ms": (final - candidate.arrival_monotonic_ns) / 1e6,
                "status": status,
                "error_code": error_code,
            }
        )
