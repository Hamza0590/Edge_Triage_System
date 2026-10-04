# Logging and run ownership

Use `EventLogger.emit` for run events and `logger.bind(trace_id, candidate_id).emit`
for candidate events. Bindings cannot override their correlation IDs. All events
include schema_version, timestamp_utc, monotonic_ns, level, run_id, trace_id,
candidate_id, module, event_type, config_hash, message, reason_code, duration_ms,
payload, error_type and error_message. Run-level candidate/trace IDs may be null.

| Event family | Intended producer |
|---|---|
| run.* | Lifecycle and reproducibility |
| data.* | Audit, preprocessing, split and candidate references |
| model.* | Model construction, training and checkpoints |
| monitor.* | Resource sampling |
| scheduler.* | Tier/admission/concurrency decisions |
| inference.* | Executor dispatch and prediction |
| uncertainty.* | MC passes and escalation |
| triage.* | Threshold application and fate |
| experiment.* | Workloads, ablations and aggregation |
| error.* | Bounded failures |

Each event is a flushed UTF-8 JSONL record plus a concise stderr message. One logger
owns each run's stream and serializes concurrent thread writes; multi-process
executors must forward events to this owner. Exceptions store type and redacted
message, never stack traces. Disk failures propagate; console failures do not lose
the persisted event.

Payloads accept JSON primitives, enums, dataclasses and validated models. Strings
are capped at 1,024 characters, mappings/sequences at 64 entries, nesting at eight
levels, and encoded payloads at 8,192 bytes (oversized payloads become a marker).
Known secret-like keys are recursively redacted; recognizable key=value and Bearer
patterns are redacted in text. This does not detect arbitrary secrets in prose:
callers must not supply secrets. Image/tensor keys, arrays, bytes and unsupported
objects are rejected, never stringified or converted to lists.

RunManager verifies raw hashes before creating a unique run. It writes the resolved
configuration and initial manifest, emits run.started, and finalizes on context exit.
Successful completion produces run.completed; exceptions produce error.run_failed
and run.failed. Manifest replacement is atomic and flushed before replacement.
Finalized runs cannot be reopened through the logging API, finalized twice, or
modified through RunManager. `append_correction` writes only corrections.jsonl,
including author, reason, text and the original manifest hash. Manual filesystem
modification is outside this API-level guarantee.

`candidates.parquet` will be produced by later pipeline modules; no empty placeholder
masquerades as analytical data. Metrics may be CSV/JSON under metrics/. Configured
split, threshold and trace artifacts are SHA-256 hashed when supplied; checkpoint
hashes remain empty until models exist.
