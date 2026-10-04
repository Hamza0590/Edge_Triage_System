# Module 03 reference pipeline

Activate `Edge_Project` and run:

```powershell
edge-triage pipeline smoke --config configs/experiments/smoke.yaml --limit 20
```

The default configuration includes Tiny/Medium/Large with escalation disabled.
Six sibling `smoke_<portfolio>.yaml` configurations cover every singleton and pair.
All use real validation rows and frozen Module 02 normalization. Every manifest,
analytical record, Parquet schema and summary is marked `NON_SCIENTIFIC_SMOKE_TEST`.
Module 04 migrated these configurations to trained, hash-verified CNN checkpoints.
Hash-derived dummy adapters now exist only in `tests/fixtures/dummy.py`; the runtime
does not register a dummy backend. Smoke runs still do not provide calibrated
triage or hardware-performance evidence. See [models](models.md) for validation results.

## Components and records

`runtime/interfaces.py` defines the 11 protocol boundaries. `runtime/factory.py`
contains separate typed registries for tier selection, admission, executor, model,
uncertainty, escalation, thresholds and fate. Unsupported adaptive policies/backends
are rejected rather than silently mapped to reference implementations. Concrete
source/queue/executor implementations are in `runtime/components.py`.

`runtime/pipeline.py` orchestrates a finite FIFO source, independent of policy types.
It queues all supplied preprocessed items, captures CPU/RAM snapshots (GPU/network
unknown), asks for tier selection and admission, then executes and applies UQ,
escalation and fate policies. The snapshot's arrival rate is a reference placeholder
zero; real monitoring and streaming arrivals are Module 06. Delayed admission sleeps
for a configurable interval up to a monotonic deadline. Rejection or timeout produces
a failed candidate record, not a silent discard. Later candidates continue.

Detailed execution state is authoritative in `runtime/state.py`, not the input
Candidate's immutable submission-state field. Escalation returns through scheduling
and admission before re-executing. Invalid state edges produce an error event and a
typed exception. Every event retains the original run/trace/candidate identity.
`EventLogger.append` accepts validated events while preserving their timestamps and
applying the same bounded/redacted payload rules. `LoggerEventSink` bridges protocols.

`runtime/records.py` defines `CandidateRecord` and its fixed flattened Arrow schema.
One row is persisted for each completed/failed candidate. Columns contain identity,
separate offline label, UTC arrival/final strings, monotonic stage timestamps, tier
path, initial/final probabilities/logits, uncertainty summary, resource snapshot JSON,
queue/concurrency counts, fate, exact threshold identity, byte counts, durations,
status and error code. No tensor, image, binary or array columns exist. Stage UTC
timestamps are available in correlated JSONL events.

Parquet batches flush at the configured record count and on normal shutdown; empty
runs still have a readable schema. Candidate exceptions are isolated. Persistence
errors remain run failures rather than claims of successful logging. The sequential
executor and Parquet writer close in `finally`, before the run manifest is finalized.
A completed run may contain failed candidates; inspect `metrics/smoke_summary.json`.
Abrupt process termination is not a transactional checkpoint guarantee.

## Timing and semantics

- Queue delay is first execution start minus queue entry (or failure completion if
  never admitted). It includes scheduling/admission overhead before execution.
- Inference time sums measured executor-call durations, including failed calls.
- End-to-end time is final minus source arrival; intermediate decisions/logging and
  admission waits are included. Final-record persistence follows this timestamp.
- Source arrival occurs after the offline dataset has applied frozen preprocessing;
  this smoke pipeline does not claim end-to-end physical sensor latency.
- Common threshold 0.5 is explicitly uncalibrated. Its exact textual definition is
  persisted and SHA-256 hashed in both the manifest and every fate record.
- ClassificationOnlyFatePolicy records Real/Bogus classification reasons while
  transmitting all original bytes. No compression or recall-constrained discards occur.
- Analytical schema v1 adds an optional preprocessing hash to the run manifest;
  old manifests remain valid. Split and preprocessing hashes are bound before the run.

## Research provenance

This is project-specific engineering integration. Source relationships below are
**inspired at interface stage**, as directed by the implementation brief. No source
algorithm, optimized scheduler or theoretical guarantee is reproduced. Direct DOI
fetches were unavailable during this implementation; scientific-method verification
remains required in Modules 07/09. Code is locally authored; upstream licenses and
code availability remain `unknown_pending_verification`.

| Source | Interface principle | Deviations and excluded features | Verification evidence |
|---|---|---|---|
| [Taylor et al.](https://doi.org/10.1145/3211332.3211336) | Portfolio and selection boundary | Fixed tier; no adaptive learned selector | All seven portfolio tests and selector substitution |
| [PRAS / Sun et al.](https://doi.org/10.1016/j.sysarc.2025.103597) | Decisions after arrival/queue entry | Finite FIFO; no paper scheduler or optimized pipeline | Stage/correlation/timing tests |
| [Quilt / Lee et al.](https://doi.org/10.1016/j.sysarc.2026.103696) | Admission distinct from tier selection | Sequential capacity only; no model partitioning or calibrated memory policy | Admission delay, timeout and independent-policy tests |

Parquet persistence uses [Apache Arrow's writer API](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html),
validated against installed PyArrow 23.0.1. No scientific results are claimed here.
