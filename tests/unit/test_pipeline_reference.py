import pytest
import torch

from edge_triage.contracts import Candidate, ModelTier, ResourceSnapshot, TierJobs, utc_now
from edge_triage.runtime.interfaces import CandidateInput
from edge_triage.runtime.reference import (
    ClassificationOnlyFatePolicy,
    CommonThresholdPolicy,
    NoEscalationPolicy,
    NoUncertaintyPolicy,
    SequentialAdmissionPolicy,
    StaticTierPolicy,
)
from edge_triage.runtime.state import InvalidTransitionError, PipelineState, StateMachine
from tests.fixtures.dummy import DummyModelAdapter


class MemorySink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def candidate():
    return Candidate(
        run_id="smoke", trace_id="trace", candidate_id="c1", dataset_index=0, arrival_monotonic_ns=0
    )


def resources(active=0):
    return ResourceSnapshot(
        timestamp_utc=utc_now(),
        monotonic_ns=1,
        cpu_percent=0,
        available_ram_bytes=1024,
        queue_length=1,
        arrival_rate_per_s=0,
        active_jobs_by_tier=(TierJobs(tier=ModelTier.TINY, count=active),),
    )


def test_complete_state_path_and_correlation():
    sink = MemorySink()
    machine = StateMachine(candidate(), "a" * 64, sink)
    for state in (
        PipelineState.QUEUED,
        PipelineState.SCHEDULED,
        PipelineState.ADMITTED,
        PipelineState.RUNNING,
        PipelineState.PREDICTED,
        PipelineState.UNCERTAINTY,
        PipelineState.ESCALATED,
        PipelineState.SCHEDULED,
        PipelineState.ADMITTED,
        PipelineState.RUNNING,
        PipelineState.PREDICTED,
        PipelineState.TRIAGED,
        PipelineState.COMPLETED,
    ):
        machine.transition(state)
    assert machine.state == PipelineState.COMPLETED
    assert all(
        (event.run_id, event.trace_id, event.candidate_id) == ("smoke", "trace", "c1")
        for event in sink.events
    )
    with pytest.raises(InvalidTransitionError):
        machine.transition(PipelineState.RUNNING)
    assert sink.events[-1].event_type == "error.invalid_transition"
    assert machine.state == PipelineState.COMPLETED


def test_invalid_transition_is_logged_without_mutating_state():
    sink = MemorySink()
    machine = StateMachine(candidate(), "a" * 64, sink)
    with pytest.raises(InvalidTransitionError):
        machine.transition(PipelineState.COMPLETED)
    assert machine.state == PipelineState.ARRIVED
    assert sink.events[0].reason_code == "INVALID_STATE_TRANSITION"
    machine.transition(PipelineState.FAILED)
    with pytest.raises(InvalidTransitionError):
        machine.transition(PipelineState.QUEUED)


@pytest.mark.parametrize("tier", list(ModelTier))
def test_reference_policies_are_explicit_and_ignore_ground_truth(tier):
    item = CandidateInput(candidate(), torch.zeros(3, 30, 30), 0, 120000)
    other = CandidateInput(candidate(), torch.ones(3, 30, 30), 1, 120000)
    model = DummyModelAdapter()
    prediction = model.predict(item, tier)
    assert model.predict(other, tier).logit == prediction.logit
    assert "NON_SCIENTIFIC" in prediction.checkpoint_id
    assert StaticTierPolicy(tier).select(item.candidate, resources()).selected_tier == tier
    uncertainty = NoUncertaintyPolicy().evaluate(prediction)
    assert uncertainty.method == "none" and uncertainty.variance == 0
    assert not NoEscalationPolicy().decide(prediction, uncertainty).escalate
    threshold = CommonThresholdPolicy().resolve(tier)
    fate = ClassificationOnlyFatePolicy().decide(prediction, threshold, item.original_bytes)
    assert fate.action == "transmit" and fate.input_bytes == fate.output_bytes == 120000


def test_admission_delay_is_explicit():
    policy = SequentialAdmissionPolicy()
    assert policy.decide(candidate(), resources(0)).outcome == "admitted"
    delayed = policy.decide(candidate(), resources(1))
    assert delayed.outcome == "delayed" and delayed.reason_code == "WAIT_ACTIVE_JOB"
    with pytest.raises(ValueError):
        SequentialAdmissionPolicy(2)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_threshold_validation(threshold):
    with pytest.raises(ValueError):
        CommonThresholdPolicy(threshold)
