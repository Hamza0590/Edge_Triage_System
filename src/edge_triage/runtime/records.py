"""Versioned flattened analytics; explicit schema prevents raw tensor serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import model_validator

from edge_triage.contracts import Contract, Count, NonNegative, Probability


class CandidateRecord(Contract):
    schema_version: Literal[1] = 1
    purpose: Literal["NON_SCIENTIFIC_SMOKE_TEST"] = "NON_SCIENTIFIC_SMOKE_TEST"
    run_id: str
    trace_id: str
    candidate_id: str
    label: Literal[0, 1]
    arrival_utc: str
    final_utc: str
    arrival_ns: Count
    queue_ns: Count
    scheduled_ns: Count | None
    start_ns: Count | None
    end_ns: Count | None
    final_ns: Count
    selected_tiers: str
    initial_logit: float | None
    final_logit: float | None
    initial_p_real: Probability | None
    final_p_real: Probability | None
    uncertainty_variance: NonNegative | None
    uncertainty_method: str | None
    escalation_path: str
    prediction_chain_json: str = "[]"
    uncertainty_chain_json: str = "[]"
    escalation_chain_json: str = "[]"
    final_trusted_tier: str | None = None
    final_trusted_p_real: Probability | None = None
    conservative_fallback: Literal[0, 1] = 0
    uq_ms: NonNegative = 0
    admission_ms: NonNegative = 0
    executor_ms: NonNegative = 0
    execution_chain_json: str = "[]"
    escalation_decision_ms: NonNegative = 0
    additional_forward_passes: Count = 0
    failed_uq_attempts: Count = 0
    failed_uq_completed: Count = 0
    resource_snapshot_json: str
    queue_length: Count
    active_jobs: Count
    concurrency: Count
    fate: str | None
    threshold_artifact_id: str | None
    threshold_artifact_sha256: str | None
    input_bytes: Count
    output_bytes: Count
    queue_ms: NonNegative
    inference_ms: NonNegative
    end_to_end_ms: NonNegative
    status: Literal["completed", "failed"]
    error_code: str | None

    @model_validator(mode="after")
    def validate_clocks(self) -> Self:
        clocks = [
            value
            for value in (
                self.arrival_ns,
                self.queue_ns,
                self.scheduled_ns,
                self.start_ns,
                self.end_ns,
                self.final_ns,
            )
            if value is not None
        ]
        if clocks != sorted(clocks):
            raise ValueError("candidate timestamps must be monotonic")
        if abs(self.end_to_end_ms - (self.final_ns - self.arrival_ns) / 1e6) > 1e-6:
            raise ValueError("end-to-end latency identity violated")
        if self.inference_ms > self.end_to_end_ms:
            raise ValueError("inference exceeds end-to-end latency")
        if (self.status == "failed") != (self.error_code is not None):
            raise ValueError("only failed candidates carry an error code")
        return self


_INTEGERS = {
    "failed_uq_attempts",
    "failed_uq_completed",
    "conservative_fallback",
    "additional_forward_passes",
    "schema_version",
    "label",
    "arrival_ns",
    "queue_ns",
    "scheduled_ns",
    "start_ns",
    "end_ns",
    "final_ns",
    "queue_length",
    "active_jobs",
    "concurrency",
    "input_bytes",
    "output_bytes",
}
_FLOATS = {
    "admission_ms",
    "executor_ms",
    "final_trusted_p_real",
    "uq_ms",
    "escalation_decision_ms",
    "initial_logit",
    "final_logit",
    "initial_p_real",
    "final_p_real",
    "uncertainty_variance",
    "queue_ms",
    "inference_ms",
    "end_to_end_ms",
}
ANALYTICAL_SCHEMA = pa.schema(
    [
        pa.field(
            name,
            pa.int64() if name in _INTEGERS else pa.float64() if name in _FLOATS else pa.string(),
        )
        for name in CandidateRecord.model_fields
    ],
    metadata={b"schema_version": b"1", b"purpose": b"NON_SCIENTIFIC_SMOKE_TEST"},
)


class ParquetRecordWriter:
    """Run-owned writer; close flushes partial batches even on upstream failures."""

    def __init__(self, path: Path, batch_size: int = 64) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("xb")
        self._writer = pq.ParquetWriter(self._stream, ANALYTICAL_SCHEMA)
        self._buffer: list[CandidateRecord] = []
        self.batch_size = batch_size
        self.closed = False
        self.count = 0

    def append(self, record: CandidateRecord) -> None:
        if self.closed:
            raise RuntimeError("analytical writer is closed")
        self._buffer.append(record)
        self.count += 1
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            self._writer.write_table(
                pa.Table.from_pylist(
                    [record.model_dump(mode="json") for record in self._buffer],
                    schema=ANALYTICAL_SCHEMA,
                )
            )
            self._buffer.clear()

    def close(self) -> None:
        if not self.closed:
            try:
                self.flush()
            finally:
                self._writer.close()
                self._stream.close()
                self.closed = True
