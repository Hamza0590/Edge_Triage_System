"""Verify Module07 with frozen trace replay; no trace generation or test-set evaluation."""

import argparse
import contextlib
import io
import itertools
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from preflight_module07 import verify as preflight

from edge_triage.calibration.profiling import hardware_identity
from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import CandidateEvent, ModelTier, ResourceSnapshot
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import dataset_hashes, hash_file
from edge_triage.runtime.streaming import run_adaptive_sequential


def verify(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = preflight()
    config = load_config(root / "configs/experiments/module07_v1.yaml")
    actual_hardware, _ = hardware_identity()
    assert actual_hardware == config.scheduler.hardware_id
    suite = json.loads((root / "artifacts/traces/suite_a515d472c9d5ef05.json").read_text())
    backlog = Path(
        next(
            e["path"]
            for e in suite["traces"]
            if e["profile"] == "backlog" and e["process"] == "periodic"
        )
    )
    runs = []
    portfolios = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]
    jobs = [(p, backlog) for p in portfolios] + [(tuple(ModelTier), config.workload.trace)]
    for portfolio, trace in jobs:
        raw = config.model_dump()
        raw["models"]["tier_portfolio"] = portfolio
        raw["workload"]["trace"] = trace
        current = AppConfig.model_validate(raw)
        with contextlib.redirect_stderr(io.StringIO()):
            directory = run_adaptive_sequential(current)
        summary_path = directory / "metrics/adaptive_sequential_summary.json"
        summary = json.loads(summary_path.read_text())
        rows = pq.read_table(directory / "candidates.parquet").to_pylist()
        expected = load_trace(trace, hash_file(config.split.manifest))
        assert summary["completed"] == len(expected) and summary["failed"] == 0, directory
        assert {r["candidate_id"] for r in rows} == {r.candidate_id for r in expected}
        assert all(r["selected_tiers"] in {t.value for t in portfolio} for r in rows)
        assert all(r["concurrency"] == 1 for r in rows)
        assert (
            sum(
                summary["queue"][k]
                for k in ("rejected", "dropped", "cancelled", "active", "length")
            )
            == 0
        )
        resources = [
            ResourceSnapshot.model_validate_json(line)
            for line in (directory / "resources.jsonl").read_text().splitlines()
        ]
        ids = {s.snapshot_id for s in resources}
        events = [
            CandidateEvent.model_validate_json(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        decisions = [e for e in events if e.event_type == "scheduler.decision"]
        assert len(decisions) == 2 * len(expected)
        for event in decisions:
            detail = event.payload["explanation"]
            assert detail["snapshot_id"] in ids
            assert detail["portfolio"] == [t.value for t in portfolio]
            assert event.duration_ms is not None and event.duration_ms >= 0
            assert "TRUNCATED" not in event.model_dump_json()
        result = {
            "run": str(directory),
            "portfolio": [t.value for t in portfolio],
            "summary": summary,
            "summary_sha256": hash_file(summary_path),
            "manifest_sha256": hash_file(directory / "manifest.json"),
            "events_sha256": hash_file(directory / "events.jsonl"),
            "selected_tiers": dict(Counter(r["selected_tiers"] for r in rows)),
            "reason_codes": dict(Counter(e.reason_code for e in decisions)),
        }
        runs.append(result)
        print(f"{directory.name}: {len(rows)} completed; {result['selected_tiers']}", flush=True)
    overhead = output.parent / "overhead.json"
    assert overhead.exists()
    write_immutable(
        output,
        canonical(
            {
                "schema_version": 1,
                "verified": True,
                "preflight": evidence,
                "runs": runs,
                "hardware_id": actual_hardware,
                "overhead_sha256": hash_file(overhead),
                "dataset_hashes": [a.model_dump(mode="json") for a in dataset_hashes(config)],
                "test_evaluated": False,
                "classification_metrics": False,
            }
        ),
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
