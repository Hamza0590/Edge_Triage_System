"""Non-scientific instrumentation benchmark; dummy remains in test fixtures only."""

from __future__ import annotations

import argparse
import io
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_triage.calibration.profiling import summarize  # noqa: E402
from edge_triage.config import load_config  # noqa: E402
from edge_triage.contracts import Candidate  # noqa: E402
from edge_triage.data.artifacts import canonical, write_immutable  # noqa: E402
from edge_triage.event_logging import EventLogger  # noqa: E402
from edge_triage.health import hash_file  # noqa: E402
from edge_triage.runtime.factory import MODELS, assemble  # noqa: E402
from edge_triage.runtime.interfaces import CandidateInput  # noqa: E402
from edge_triage.runtime.pipeline import Pipeline  # noqa: E402
from edge_triage.runtime.records import ParquetRecordWriter  # noqa: E402
from tests.fixtures.dummy import DummyModelAdapter  # noqa: E402


def benchmark(output: Path, count: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/smoke_tiny.yaml")
    config = config.model_copy(
        update={"pipeline": config.pipeline.model_copy(update={"model_adapter": "dummy"})}
    )
    MODELS["dummy"] = lambda _: DummyModelAdapter()
    image = torch.zeros(3, 30, 30)
    rounds = []
    for repeat in range(3):  # First full round is the warm-up, retained for audit.
        run_id = f"overhead-{output.parent.name}-{repeat}"
        directory = output.parent / f"overhead_round{repeat}"
        directory.mkdir(parents=True, exist_ok=False)

        def source() -> Iterator[CandidateInput]:
            for index in range(count):
                yield CandidateInput(
                    Candidate(
                        run_id=run_id,
                        trace_id=f"{run_id}:{index}",
                        candidate_id=f"synthetic-{index}",
                        dataset_index=index,
                        arrival_monotonic_ns=time.monotonic_ns(),
                    ),
                    image,
                    index % 2,
                    120000,
                )

        console = io.StringIO()
        logger = EventLogger(directory / "events.jsonl", run_id, config.config_hash, console)
        writer = ParquetRecordWriter(directory / "candidates.parquet", 64)
        parts = assemble(config)
        threshold = directory / "common_threshold.txt"
        write_immutable(threshold, parts.thresholds.definition())  # type: ignore[attr-defined]
        start = time.perf_counter_ns()
        completed, failed = Pipeline(config, parts, logger, writer).run(source())
        logger.close()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        records = pq.read_table(directory / "candidates.parquet").to_pylist()
        if completed != count or failed or len(records) != count:
            raise RuntimeError("overhead benchmark candidate failure")
        rounds.append(
            {
                "warmup": repeat == 0,
                "count": count,
                "wall_ms": elapsed,
                "wall_ms_per_candidate": elapsed / count,
                "candidate_end_to_end": summarize(
                    [r["end_to_end_ms"] for r in records]
                ).model_dump(),
                "executor_ms_mean": sum(r["inference_ms"] for r in records) / count,
                "artifacts": {
                    name: hash_file(directory / name)
                    for name in ("events.jsonl", "candidates.parquet", "common_threshold.txt")
                },
            }
        )
    # Isolate logger serialization/write cost; in-memory console avoids terminal rendering noise.
    logger = EventLogger(
        output.parent / "logger_only.jsonl", "logger-only", config.config_hash, io.StringIO()
    )
    times = []
    for index in range(120):
        start = time.perf_counter_ns()
        logger.emit(
            module="profiling",
            event_type="experiment.overhead",
            reason_code="NON_SCIENTIFIC",
            payload={"sequence": index},
        )
        if index >= 20:
            times.append((time.perf_counter_ns() - start) / 1e6)
    logger.close()
    result = {
        "purpose": "NON_SCIENTIFIC_INSTRUMENTATION_BENCHMARK",
        "rounds": rounds,
        "scope": "synthetic source+FIFO+reference policies+minimal monitor+dummy+"
        "JSONL+Parquet incl final flush; excludes run setup and model loading",
        "console": "StringIO human-readable echo; terminal rendering excluded",
        "logger_only": summarize(times).model_dump(),
        "logger_only_sha256": hash_file(output.parent / "logger_only.jsonl"),
        "subtraction_warning": "report overhead separately; not an additive causal estimate",
    }
    write_immutable(output, canonical(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    benchmark(args.output, args.count)
