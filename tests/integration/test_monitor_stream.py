import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ResourceSnapshot
from edge_triage.experiments.workloads import load_trace, persist_trace
from edge_triage.health import hash_file
from edge_triage.runtime.streaming import run_monitor_only

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("inject_failure", [False, True])
def test_monitor_only_real_stream_no_loss_and_snapshot_references(
    tmp_path, monkeypatch, inject_failure
):
    class Unavailable:
        def __init__(self, *args):
            pass

        def collect(self):
            return {"availability": {"gpu": "unavailable:test"}}

    monkeypatch.setattr("edge_triage.runtime.streaming.NvidiaCollector", Unavailable)
    if inject_failure:
        from edge_triage.runtime import streaming

        original = streaming.assemble

        class FailOne:
            def __init__(self, base):
                self.base = base

            def predict(self, item, tier):
                if item.candidate.candidate_id.endswith("occurrence-5"):
                    raise RuntimeError("injected inference failure")
                return self.base.predict(item, tier)

        def assemble(*args, **kwargs):
            parts = original(*args, **kwargs)
            parts.models = {tier: FailOne(model) for tier, model in parts.models.items()}
            return parts

        monkeypatch.setattr(streaming, "assemble", assemble)
    raw = load_config(ROOT / "configs/experiments/module06_v1.yaml").model_dump()
    raw["paths"].update(runs_dir=tmp_path / "runs", artifacts_dir=tmp_path / "artifacts")
    raw["queue"]["capacity"] = 1
    raw["monitor"]["interval_s"] = 0.005
    raw["workload"].update(count=24, profile="backlog")
    config = AppConfig.model_validate(raw)
    trace = persist_trace(config, config.workload)
    assert persist_trace(config, config.workload) == trace
    rows = load_trace(trace, hash_file(config.split.manifest))
    config = config.model_copy(
        update={"workload": config.workload.model_copy(update={"trace": trace})}
    )
    directory = run_monitor_only(config)
    summary = json.loads((directory / "metrics/monitor_only_summary.json").read_text())
    assert summary["completed"] == 24 - int(inject_failure)
    assert summary["failed"] == int(inject_failure)
    assert summary["queue"]["peak_length"] == 1
    assert summary["queue"]["blocked_puts"] > 0
    assert (
        summary["queue"]["rejected"]
        == summary["queue"]["dropped"]
        == summary["queue"]["cancelled"]
        == 0
    )
    assert summary["scheduler"]["calls"] == 24
    assert summary["scheduler"]["admission_calls"] == 24
    assert summary["monitor"]["samples"] >= 2
    assert summary["trace_sha256"] == hash_file(trace)
    records = pq.read_table(directory / "candidates.parquet").to_pylist()
    assert {r["candidate_id"] for r in records} == {r.candidate_id for r in rows}
    resources = [
        ResourceSnapshot.model_validate_json(line)
        for line in (directory / "resources.jsonl").read_text().splitlines()
    ]
    references = {s.snapshot_id for s in resources}
    for r in records:
        for snapshot in json.loads(r["resource_snapshot_json"]):
            if "snapshot_id" in snapshot and snapshot["snapshot_id"] is not None:
                assert snapshot["snapshot_id"] in references
                assert "cpu_percent" not in snapshot
    assert summary["test_evaluated"] is False
    trace.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_trace(trace, hash_file(config.split.manifest))
