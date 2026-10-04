"""Single structured logging API with bounded, redacted metadata payloads."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from edge_triage.contracts import CandidateEvent, utc_now

MAX_PAYLOAD_BYTES = 8192
SECRET_KEY = re.compile(
    r"password|passwd|secret|token|api.?key|authorization|cookie|credential", re.I
)
SECRET_TEXT = re.compile(
    r"(?i)((?:password|passwd|secret|token|api[_-]?key|authorization|credential)"
    r"\s*[=:]\s*)(?:Bearer\s+)?[^\s,;]+"
)
BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
RAW_KEYS = {"image", "images", "tensor", "tensors", "pixels", "raw_image", "raw_tensor"}


def safe_text(value: str, limit: int = 1024) -> str:
    value = SECRET_TEXT.sub(r"\1[REDACTED]", value)
    value = BEARER.sub("Bearer [REDACTED]", value)
    return value[:limit]


def sanitize(value: Any, depth: int = 0) -> Any:
    """Allow only bounded JSON metadata; reject arrays, bytes and arbitrary objects.

    Tensor libraries are deliberately not imported by the logger.
    """
    if depth > 8:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, Enum):
        return sanitize(value.value, depth + 1)
    if isinstance(value, str):
        return safe_text(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return safe_text(str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite payload value")
        return value
    if isinstance(value, BaseModel):
        return sanitize(
            {name: getattr(value, name) for name in type(value).model_fields}, depth + 1
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize(
            {field.name: getattr(value, field.name) for field in dataclasses.fields(value)},
            depth + 1,
        )
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 64:
                result["_truncated"] = True
                break
            if not isinstance(key, str):
                raise TypeError("payload keys must be strings")
            if SECRET_KEY.search(key):
                result[safe_text(key, 128)] = "[REDACTED]"
            elif key.lower() in RAW_KEYS:
                raise TypeError("raw image/tensor payloads are prohibited")
            else:
                result[safe_text(key, 128)] = sanitize(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize(item, depth + 1) for item in value[:64]]
    raise TypeError("unsupported payload type; tensors/images/binary objects are prohibited")


class EventLogger:
    """Thread-safe run-owned append-only sink; close before sealing the manifest.

    Disk write errors propagate so callers cannot silently claim durable results.
    Exception messages are redacted and bounded; tracebacks are never serialized.
    """

    def __init__(
        self, path: Path, run_id: str, config_hash: str, console: TextIO | None = None
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.config_hash = config_hash
        self.console = console if console is not None else sys.stderr
        self._lock = threading.RLock()
        self._check_unsealed()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", newline="\n")

    def _check_unsealed(self) -> None:
        manifest = self.path.parent / "manifest.json"
        if manifest.exists():
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if record["status"] != "running":
                raise RuntimeError("completed/failed run is immutable")

    def emit(
        self,
        *,
        module: str,
        event_type: str,
        reason_code: str,
        message: str = "",
        trace_id: str | None = None,
        candidate_id: str | None = None,
        level: str = "INFO",
        duration_ms: float | None = None,
        payload: Any = None,
        exception: BaseException | None = None,
    ) -> CandidateEvent:
        cleaned = sanitize({} if payload is None else payload)
        if not isinstance(cleaned, dict):
            raise TypeError("event payload must be a mapping or dataclass/model")
        if len(json.dumps(cleaned, ensure_ascii=False).encode("utf-8")) > MAX_PAYLOAD_BYTES:
            cleaned = {"_truncated": True, "reason": "payload_byte_limit"}
        event = CandidateEvent.model_validate(
            {
                "timestamp_utc": utc_now(),
                "monotonic_ns": time.monotonic_ns(),
                "level": level,
                "run_id": self.run_id,
                "trace_id": trace_id,
                "candidate_id": candidate_id,
                "module": module,
                "event_type": event_type,
                "config_hash": self.config_hash,
                "message": safe_text(message),
                "reason_code": reason_code,
                "duration_ms": duration_ms,
                "payload": cleaned,
                "error_type": type(exception).__name__[:256] if exception else None,
                "error_message": safe_text(str(exception)) if exception else None,
            }
        )
        return self.append(event)

    def append(self, event: CandidateEvent) -> CandidateEvent:
        """Append a typed event without replacing its original clocks or correlation IDs."""
        if event.run_id != self.run_id or event.config_hash != self.config_hash:
            raise ValueError("event belongs to another run or configuration")
        cleaned = sanitize(event.payload)
        if len(json.dumps(cleaned, ensure_ascii=False).encode("utf-8")) > MAX_PAYLOAD_BYTES:
            cleaned = {"_truncated": True, "reason": "payload_byte_limit"}
        event = CandidateEvent.model_validate(
            {
                **event.model_dump(),
                "payload": cleaned,
                "message": safe_text(event.message),
                "error_message": safe_text(event.error_message) if event.error_message else None,
            }
        )
        with self._lock:
            self._check_unsealed()
            self._stream.write(event.model_dump_json() + "\n")
            self._stream.flush()
            # The console is best effort; a broken pipe must not lose durable events.
            try:
                self.console.write(f"{event.level} {event.event_type}: {event.message}\n")
                self.console.flush()
            except (OSError, ValueError, UnicodeError):
                pass
        return event

    def bind(self, trace_id: str, candidate_id: str) -> BoundLogger:
        return BoundLogger(self, trace_id, candidate_id)

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.flush()
                self._stream.close()

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclasses.dataclass(frozen=True)
class BoundLogger:
    sink: EventLogger
    trace_id: str
    candidate_id: str

    def emit(self, **kwargs: Any) -> CandidateEvent:
        if "trace_id" in kwargs or "candidate_id" in kwargs:
            raise ValueError("bound correlation IDs cannot be overridden")
        return self.sink.emit(trace_id=self.trace_id, candidate_id=self.candidate_id, **kwargs)
