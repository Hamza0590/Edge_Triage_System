"""Detailed execution lifecycle, separate from the coarse Candidate work state."""

from __future__ import annotations

import time
from enum import Enum

from edge_triage.contracts import Candidate, CandidateEvent, utc_now
from edge_triage.runtime.interfaces import EventSink


class PipelineState(str, Enum):
    ARRIVED = "arrived"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    ADMITTED = "admitted"
    RUNNING = "running"
    PREDICTED = "predicted"
    UNCERTAINTY = "uncertainty"
    ESCALATED = "escalated"
    TRIAGED = "triaged"
    COMPLETED = "completed"
    FAILED = "failed"


class InvalidTransitionError(ValueError):
    """An invalid lifecycle edge; the attempted transition is also emitted."""


_ALLOWED = {
    PipelineState.ARRIVED: {PipelineState.QUEUED},
    PipelineState.QUEUED: {PipelineState.SCHEDULED},
    PipelineState.SCHEDULED: {PipelineState.ADMITTED},
    PipelineState.ADMITTED: {PipelineState.RUNNING},
    PipelineState.RUNNING: {PipelineState.PREDICTED},
    PipelineState.PREDICTED: {
        PipelineState.UNCERTAINTY,
        PipelineState.ESCALATED,
        PipelineState.TRIAGED,
    },
    PipelineState.UNCERTAINTY: {PipelineState.ESCALATED, PipelineState.TRIAGED},
    PipelineState.ESCALATED: {PipelineState.SCHEDULED},
    PipelineState.TRIAGED: {PipelineState.COMPLETED},
}


class StateMachine:
    """Escalation re-enters scheduling/admission before another inference."""

    def __init__(self, candidate: Candidate, config_hash: str, sink: EventSink) -> None:
        self.candidate = candidate
        self.config_hash = config_hash
        self.sink = sink
        self._state = PipelineState.ARRIVED

    @property
    def state(self) -> PipelineState:
        return self._state

    def transition(self, target: PipelineState) -> None:
        terminal = self._state in (PipelineState.COMPLETED, PipelineState.FAILED)
        valid = not terminal and (target == PipelineState.FAILED or target in _ALLOWED[self._state])
        event = CandidateEvent(
            timestamp_utc=utc_now(),
            monotonic_ns=time.monotonic_ns(),
            level="INFO" if valid else "ERROR",
            run_id=self.candidate.run_id,
            trace_id=self.candidate.trace_id,
            candidate_id=self.candidate.candidate_id,
            module="runtime.state",
            event_type="inference.state" if valid else "error.invalid_transition",
            config_hash=self.config_hash,
            message=f"{self._state.value} -> {target.value}",
            reason_code="STATE_TRANSITION" if valid else "INVALID_STATE_TRANSITION",
            duration_ms=None,
            payload={"from_state": self._state.value, "to_state": target.value},
            error_type=None if valid else "InvalidTransitionError",
            error_message=None if valid else "illegal pipeline lifecycle edge",
        )
        self.sink.emit(event)
        if not valid:
            raise InvalidTransitionError(event.message)
        self._state = target
