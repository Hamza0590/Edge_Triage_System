import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from edge_triage.contracts import CandidateEvent, ModelTier
from edge_triage.event_logging import MAX_PAYLOAD_BYTES, EventLogger


def emit(logger, **kwargs):
    return logger.emit(
        module="test", event_type="inference.completed", reason_code="TEST", **kwargs
    )


def test_schema_correlation_ids_and_concurrent_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    console = io.StringIO()
    with EventLogger(path, "r1", "a" * 64, console) as logger:
        bound = logger.bind("t1", "c1")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: emit(bound, payload={"i": i}), range(50)))
        with pytest.raises(ValueError):
            emit(bound, candidate_id="other")
    records = [
        CandidateEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 50
    assert {record.payload["i"] for record in records} == set(range(50))
    assert all(
        (record.run_id, record.trace_id, record.candidate_id) == ("r1", "t1", "c1")
        for record in records
    )
    assert all(record.schema_version == 1 and record.monotonic_ns > 0 for record in records)
    assert "INFO inference.completed" in console.getvalue()


def test_redaction_bounded_exception_and_logger_continues(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLogger(path, "r1", "a" * 64, io.StringIO()) as logger:
        event = emit(
            logger,
            message="token=supersecret Bearer abcdef",
            payload={"nested": {"api_key": "private", "password": "hidden"}},
            exception=RuntimeError("password=oops " + "x" * 2000),
        )
        assert len(event.error_message) <= 1024
        emit(logger, message="still alive")
    text = path.read_text(encoding="utf-8")
    for secret in ("supersecret", "abcdef", "private", "hidden", "oops"):
        assert secret not in text
    assert len(text.splitlines()) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"image": [[1, 2]]},
        {"nested": {"tensor": [0]}},
        {"value": b"binary"},
        {"value": object()},
        {"bad": float("nan")},
        {"bad": float("inf")},
    ],
)
def test_rejects_unsafe_payload_without_corrupting_log(tmp_path, payload):
    path = tmp_path / "events.jsonl"
    with EventLogger(path, "r1", "a" * 64, io.StringIO()) as logger:
        with pytest.raises((TypeError, ValueError)):
            emit(logger, payload=payload)
        emit(logger, message="valid")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_array_like_object_is_never_converted(tmp_path):
    class TensorLike:
        def tolist(self):
            raise AssertionError("must not materialize raw data")

        def __str__(self):
            raise AssertionError("must not stringify raw data")

    with EventLogger(tmp_path / "events.jsonl", "r1", "a" * 64, io.StringIO()) as logger:
        with pytest.raises(TypeError):
            emit(logger, payload={"value": TensorLike()})


def test_enum_dataclass_payload_limits_and_append(tmp_path):
    @dataclass
    class Metadata:
        tier: ModelTier
        count: int

    path = tmp_path / "events.jsonl"
    for _ in range(2):
        with EventLogger(path, "r1", "a" * 64, io.StringIO()) as logger:
            event = emit(logger, payload=Metadata(ModelTier.TINY, 3))
            assert event.payload["tier"] == "tiny"
            event = emit(logger, payload={str(i): "x" * 1024 for i in range(64)})
            assert len(json.dumps(event.model_dump(mode="json")["payload"])) <= MAX_PAYLOAD_BYTES
            assert event.payload["_truncated"] is True
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4


def test_broken_console_does_not_break_jsonl(tmp_path):
    console = io.StringIO()
    console.close()
    path = tmp_path / "events.jsonl"
    with EventLogger(path, "r1", "a" * 64, console) as logger:
        emit(logger)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
