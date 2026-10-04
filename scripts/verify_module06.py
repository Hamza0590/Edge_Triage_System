"""Generate/replay frozen system traces and verify complete candidate accounting."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from edge_triage.config import load_config
from edge_triage.contracts import CandidateEvent, ResourceSnapshot
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.workloads import generate_suite, load_trace, persist_trace
from edge_triage.health import dataset_hashes, hash_file
from edge_triage.runtime.streaming import run_monitor_only


def verify(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/module06_v1.yaml")
    suite = generate_suite(config)
    assert generate_suite(config) == suite
    entries = json.loads(suite.read_text())["traces"]
    assert len(entries) == 12
    for entry in entries:
        assert hash_file(Path(entry["path"])) == entry["sha256"]
        assert len(load_trace(Path(entry["path"]), hash_file(config.split.manifest))) == 120
    burst = next(
        Path(e["path"]) for e in entries if e["profile"] == "burst" and e["process"] == "periodic"
    )
    repeated = persist_trace(
        config, config.workload.model_copy(update={"profile": "backlog", "count": 700})
    )
    results = []
    for trace in (burst, repeated):
        current = config.model_copy(
            update={"workload": config.workload.model_copy(update={"trace": trace})}
        )
        run = run_monitor_only(current)
        summary = json.loads((run / "metrics/monitor_only_summary.json").read_text())
        records = pq.read_table(run / "candidates.parquet").to_pylist()
        expected = load_trace(trace, hash_file(config.split.manifest))
        assert summary["completed"] == len(expected) and summary["failed"] == 0
        assert {r["candidate_id"] for r in records} == {r.candidate_id for r in expected}
        assert summary["queue"]["peak_length"] <= config.queue.capacity
        assert summary["queue"]["length"] == summary["queue"]["active"] == 0
        assert sum(summary["queue"][key] for key in ("rejected", "dropped", "cancelled")) == 0
        snapshots = [
            ResourceSnapshot.model_validate_json(line)
            for line in (run / "resources.jsonl").read_text().splitlines()
        ]
        assert len(snapshots) >= 2
        ids = {s.snapshot_id for s in snapshots}
        events = [
            CandidateEvent.model_validate_json(line)
            for line in (run / "events.jsonl").read_text().splitlines()
        ]
        for event in events:
            if (
                event.event_type == "monitor.snapshot"
                and event.payload.get("snapshot_id") is not None
            ):
                assert event.payload["snapshot_id"] in ids
                assert "cpu_percent" not in event.payload
        assert len([e for e in events if e.event_type == "scheduler.queue.completed"]) == len(
            expected
        )
        results.append(
            {
                "run": str(run),
                "summary": summary,
                "trace": str(trace),
                "summary_sha256": hash_file(run / "metrics/monitor_only_summary.json"),
                "manifest_sha256": hash_file(run / "manifest.json"),
            }
        )
    assert results[1]["summary"]["workload"]["repeated_occurrences"] == 57
    benchmark = root / "artifacts/monitoring/module06_v1/overhead.json"
    assert len(json.loads(benchmark.read_text())["results"]) == 3
    write_immutable(
        output,
        canonical(
            {
                "schema_version": 1,
                "verified": True,
                "test_evaluated": False,
                "suite": str(suite),
                "suite_sha256": hash_file(suite),
                "runs": results,
                "overhead": str(benchmark),
                "overhead_sha256": hash_file(benchmark),
                "dataset_hashes": [a.model_dump(mode="json") for a in dataset_hashes(config)],
            }
        ),
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output)
