# Module 03 — Walking-Skeleton Pipeline and Early Scheduler Interfaces

## Mission

Build a complete but minimal end-to-end pipeline before real models and adaptive policies are added. A candidate must travel through queueing, scheduling, admission, execution, prediction, triage and structured logging using pluggable interfaces.

This module answers the architectural requirement that the scheduler exists from the start. It does not guess final resource thresholds.

## Mandatory preflight

1. Read the master index, context and Modules 01–02.
2. Verify foundation and dataset tests pass.
3. Load the frozen split and preprocessing artifacts.
4. Mark Module 03 `IN_PROGRESS`.

## Research basis and adoption

| Source | Use | Adoption |
|---|---|---|
| Taylor et al., DOI `10.1145/3211332.3211336` | Multiple pretrained-model portfolio and runtime selection abstraction | Inspired at interface stage |
| PRAS, DOI `10.1016/j.sysarc.2025.103597` | Post-arrival request/queue-aware decision boundary | Inspired at interface stage |
| Quilt, DOI `10.1016/j.sysarc.2026.103696` | Separate safe admission from model choice | Inspired at interface stage |

No claim of reproducing these schedulers is allowed in this module.

## Required pipeline interfaces

Implement small, testable protocols/abstract base classes:

```text
CandidateSource
CandidateQueue
TierSelectionPolicy
AdmissionPolicy
InferenceExecutor
ModelAdapter
UncertaintyPolicy
EscalationPolicy
ThresholdPolicy
FatePolicy
EventSink
```

Policy inputs and outputs must use the shared contracts from Module 01. Avoid passing untyped dictionaries between modules.

## Required initial policies

Implement deterministic reference policies:

- `StaticTierPolicy(tier)`
- `SequentialAdmissionPolicy(max_active=1)`
- `NoUncertaintyPolicy`
- `NoEscalationPolicy`
- `CommonThresholdPolicy`
- `ClassificationOnlyFatePolicy`

Implement a deterministic `DummyModelAdapter` whose behavior is explicit and test-only. It may derive a stable logit from candidate ID, but it must never appear in scientific results.

## Orchestrator behavior

The orchestrator must:

1. accept a candidate and preserve its correlation IDs;
2. record queue-entry time;
3. obtain a resource snapshot, initially a valid null/minimal snapshot;
4. request a tier decision;
5. request an admission decision;
6. delay without busy-spinning when not admitted;
7. execute inference;
8. optionally request uncertainty and escalation;
9. choose a fate;
10. emit a final candidate record and all intermediate events;
11. isolate one candidate failure without corrupting other candidates;
12. close cleanly and finalize the run manifest.

Use monotonic time for durations and UTC time for external event timestamps.

## Pipeline state machine

Define and validate legal states such as:

```text
ARRIVED → QUEUED → SCHEDULED → ADMITTED → RUNNING
→ PREDICTED → [UNCERTAINTY] → [ESCALATED → RUNNING → PREDICTED]
→ TRIAGED → COMPLETED
```

Any active state may transition to `FAILED`. Invalid transitions must raise a typed error and emit an error event.

## Candidate analytical record

Produce one flattened record per completed/failed candidate, including at least:

```text
candidate_id, trace_id, label, arrival/queue/start/end/final timestamps,
selected tiers, initial/final logits and p_real, uncertainty,
escalation path, resource snapshots/references, queue length,
active jobs, concurrency, fate, byte accounting,
queue/inference/end-to-end latency, status and error code
```

Write JSONL events immediately and batch analytical records to Parquet without losing records on normal shutdown.

## Configuration-driven assembly

Implement a factory/registry that assembles the pipeline from configuration. Swapping portfolio, selection policy or executor must not require editing the orchestrator.

Add smoke configurations for:

```text
tiny only + sequential
medium only + sequential
large only + sequential
tiny→medium + sequential, escalation disabled for now
tiny→large + sequential, escalation disabled for now
medium→large + sequential, escalation disabled for now
tiny→medium→large + sequential, escalation disabled for now
```

## CLI

Provide an executable smoke command similar to:

```text
edge-triage pipeline smoke --config configs/experiments/smoke.yaml --limit 20
```

It must use real dataset rows, dummy models, a sequential executor and a new run directory clearly marked `NON_SCIENTIFIC_SMOKE_TEST`.

## Required tests

- State-machine valid/invalid transition tests.
- Correlation-ID continuity test across every stage.
- Policy substitution tests proving the orchestrator does not change.
- Admission delay test without busy-looping.
- Candidate-level failure isolation test.
- Clean shutdown and record-flush test.
- Seven portfolio configuration smoke tests.
- End-to-end dummy run validating timestamps and latency identities.
- Regression test that no raw tensor is written into JSONL/Parquet.

## Acceptance criteria

- The complete logical architecture runs before real training.
- Scheduler and admission interfaces are first-class, not retrofits.
- Every tier portfolio can be assembled from configuration.
- Events explain every decision using reason codes.
- One candidate failure does not terminate the run.
- All prior tests and new tests pass.
- Context contains the stable interface registry and next action.

## Context update requirements

Register the orchestrator, state machine, policy registries, analytical schema and smoke run. Mark all dummy outputs non-scientific. Set next action to Module 04.

