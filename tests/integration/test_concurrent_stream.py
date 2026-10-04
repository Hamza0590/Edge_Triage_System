import itertools
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from edge_triage.config import AppConfig, load_config
from edge_triage.runtime.streaming import run_concurrent

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["sequential", "fixed_concurrency", "dynamic_concurrency"])
def test_real_concurrent_trace_modes(tmp_path, mode):
    config = load_config(ROOT / "configs/experiments/module09_v1.yaml")
    raw = config.model_dump()
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    raw["admission"]["execution_mode"] = mode
    raw["tier_selection"].update(policy="fixed", fixed_tier="tiny")
    suite = json.loads((ROOT / "artifacts/traces/suite_a515d472c9d5ef05.json").read_bytes())
    raw["workload"]["trace"] = Path(
        next(r["path"] for r in suite["traces"] if "backlog_periodic" in r["path"])
    )
    directory = run_concurrent(AppConfig.model_validate(raw))
    rows = pq.read_table(directory / "candidates.parquet").to_pylist()
    assert len(rows) == len({r["candidate_id"] for r in rows}) == 120
    assert all(r["status"] == "completed" and r["additional_forward_passes"] == 0 for r in rows)
    assert all(len(json.loads(r["execution_chain_json"])) == 1 for r in rows)
    summary = json.loads((directory / "metrics/concurrent_summary.json").read_bytes())
    assert summary["executor"]["peak_active"] <= (1 if mode == "sequential" else 8)
    assert summary["completed"] == 120 and summary["failed"] == 0
    assert summary["availability"]["active_jobs_total"] == 0


def test_async_common_pipeline_forced_cascades(uq_config):  # noqa: F811
    # Fixture owns synthetic routing artifact and all outputs; frozen inputs stay read-only.
    raw = uq_config.model_dump()
    raw["pipeline"]["mode"] = "concurrent"
    raw["admission"].update(policy="fixed", execution_mode="fixed_concurrency", fixed_concurrency=2)
    import asyncio
    import time

    from edge_triage.contracts import ModelTier, utc_now
    from edge_triage.runs import RunManager
    from edge_triage.runtime.components import DatasetSource, LoggerEventSink
    from edge_triage.runtime.concurrent import create_executor
    from edge_triage.runtime.factory import assemble
    from edge_triage.runtime.pipeline import Pipeline, QueueEntry, minimal_snapshot
    from edge_triage.runtime.records import ParquetRecordWriter
    from edge_triage.runtime.state import PipelineState, StateMachine

    async def execute(config):
        with RunManager(config) as run:
            parts = assemble(config, run.logger)
            writer = ParquetRecordWriter(run.directory / "candidates.parquet")
            pipeline = Pipeline(config, parts, run.logger, writer)
            engine = create_executor(config, lambda: minimal_snapshot(0))
            try:
                tasks = []
                for item in DatasetSource(config, run.manifest.run_id, 3):
                    machine = StateMachine(
                        item.candidate, run.logger.config_hash, LoggerEventSink(run.logger)
                    )
                    machine.transition(PipelineState.QUEUED)
                    entry = QueueEntry(machine, utc_now().isoformat(), time.monotonic_ns())
                    tasks.append(pipeline.process_async(item, entry, engine))
                rows = await asyncio.gather(*tasks)
                assert all(r.status == "completed" for r in rows)
                expected = "->".join(t.value for t in config.models.tier_portfolio)
                assert all(r.escalation_path == expected for r in rows)
                assert all(
                    r.additional_forward_passes == 5 * len(config.models.tier_portfolio)
                    for r in rows
                )
                for row in rows:
                    writer.append(row)
            finally:
                await engine.shutdown()
                writer.close()

    for n in (1, 2, 3):
        for portfolio in itertools.combinations(ModelTier, n):
            raw["models"]["tier_portfolio"] = portfolio
            raw["tier_selection"]["fixed_tier"] = portfolio[0]
            asyncio.run(execute(AppConfig.model_validate(raw)))


from tests.integration.test_uncertainty_cascade import uq_config  # noqa: E402,F401
