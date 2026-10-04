# Shared contracts, schema version 1

All contracts are in `edge_triage.contracts`. Pydantic models reject extra fields,
NaN/infinity and invalid field constraints; models are frozen and nested mapping
values are recursively frozen. Serialization uses `model_dump_json()` and
`Type.model_validate_json(...)`. Tensors and arbitrary Python objects have no place
in these contracts. Do not use Pydantic's unchecked `model_construct` or
`model_copy(update=...)` to bypass validation.

| Contract | Meaning and invariants |
|---|---|
| Candidate | Stable run/trace/candidate identity and nonnegative raw-array index; no pixels |
| ModelTier | Exactly tiny, medium or large |
| PredictionResult | Finite raw logit, sigmoid-derived p_real, optional calibrated_p_real with artifact ID, checkpoint/tier and timings |
| UncertaintyResult | Method/pass count, mean p_real, population variance, entropy and elapsed time |
| ResourceSnapshot | UTC/monotonic sample times, CPU/RAM/GPU, queue, arrivals, active jobs, latency, network; unavailable values are null |
| SchedulingDecision | Selected tier and nonempty reason code |
| AdmissionDecision | admitted/delayed/rejected; admission needs positive concurrency |
| EscalationDecision | Optional upward tier transition; downward/same-tier transitions forbidden |
| TriageDecision | discard/compress/transmit, exact threshold ID and SHA-256, p_real and logical byte accounting |
| CandidateEvent | Versioned bounded event with correlation IDs, timings, payload and exception metadata |
| RunManifest | Configuration/data/artifact identity, run lifecycle, environment, seeds and provenance |
| ModelProfile | Hardware/checkpoint-specific batch/concurrency, input shape, latency quantiles and peak memory |

`PredictionResult.p_bogus` is derived as `1 - p_real`; it is not independently stored.
Downstream calibrated decisions use `calibrated_p_real` when available. Variance is
the population variance of probabilities, bounded by 0.25. Durations/latencies use
milliseconds, memory uses bytes, utilization uses percentages and arrival rates
use candidates/second. All wall-clock timestamps are timezone-aware UTC.

Candidate transitions are enforced by `candidate.transition(next_state)`, returning
a new object with unchanged identity:

```text
queued -> admitted -> running -> predicted -> triaged
   |          |          |          |
rejected    failed     failed     escalated -> queued
                                    |
                                  failed
```

Delayed admission leaves the candidate queued. Rejected is distinct from scientific
discard; later modules must account for rejected real candidates when evaluating
survival recall. Terminal states cannot transition. Escalation requeues the same
candidate; trace and candidate IDs do not change.

Run state is `running -> completed|failed`. A completed/failed manifest requires an
end timestamp no earlier than its start. Artifact absence is null/empty, never an
invented hash. The run manager owns lifecycle transitions.
