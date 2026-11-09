"""Test-owned fate artifacts only; frozen research inputs remain untouched."""

import asyncio
import itertools
import json
import time
from pathlib import Path

import pytest

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier, utc_now
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.runs import RunManager
from edge_triage.runtime.components import DatasetSource, LoggerEventSink
from edge_triage.runtime.concurrent import create_executor
from edge_triage.runtime.factory import assemble, persist_thresholds
from edge_triage.runtime.pipeline import Pipeline, QueueEntry, minimal_snapshot
from edge_triage.runtime.records import ParquetRecordWriter
from edge_triage.runtime.state import PipelineState, StateMachine
from edge_triage.triage.artifacts import FateThresholdArtifact, load_fate_artifact, system_identity

ROOT = Path(__file__).resolve().parents[2]
PORTFOLIOS = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]


def configured(tmp_path, portfolio, mode, *, fate="recall_constrained", common=False):
    raw = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml").model_dump()
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    raw["models"]["tier_portfolio"] = portfolio
    raw["tier_selection"].update(policy="round_robin", fixed_tier=portfolio[0])
    raw["admission"].update(execution_mode=mode, fixed_concurrency=2)
    raw["fate"].update(policy=fate, metadata_bytes=32)
    raw["threshold"].update(
        policy="common" if common else "per_tier", artifact=None, artifact_sha256=None
    )
    config = AppConfig.model_validate(raw)
    family = json.loads(config.models.checkpoint_index.read_bytes())
    calibrations = json.loads(config.calibration.artifact_index.read_bytes())
    specs = []
    for tier in portfolio:
        cal = json.loads(Path(calibrations["calibrations"][tier.value]["path"]).read_bytes())
        calibrated = cal["recommended_mode"] == "calibrated"
        specs.append(
            dict(
                tier=tier,
                checkpoint_id=family["checkpoints"][tier.value]["checkpoint_id"],
                probability_mode=cal["recommended_mode"],
                calibration_artifact_id=cal["artifact_id"] if calibrated else None,
                operating_point=dict(max_p_real_for_discard=0.2),
            )
        )
    artifact = FateThresholdArtifact(
        artifact_id="TEST_ONLY_NOT_SCIENTIFIC",
        threshold_policy=config.threshold.policy,
        system_sha256=system_identity(config),
        family_sha256=hash_file(config.models.checkpoint_index),
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        calibration_index_sha256=hash_file(config.calibration.artifact_index),
        validation_records_sha256="0" * 64,
        target_survival_recall=0.95,
        validation_survival_recall=1,
        objective="Synthetic test fixture; not fitted research evidence",
        validation_metrics={},
        tiers=tuple(specs),
    )
    path = tmp_path / "test_threshold.json"
    write_immutable(path, canonical(artifact.model_dump(mode="json")))
    raw["threshold"].update(artifact=path, artifact_sha256=hash_file(path))
    return AppConfig.model_validate(raw)


@pytest.mark.parametrize("portfolio", PORTFOLIOS)
@pytest.mark.parametrize("mode", ["sequential", "fixed_concurrency", "dynamic_concurrency"])
def test_all_real_portfolios_modes_use_frozen_fate(tmp_path, portfolio, mode):
    config = configured(tmp_path, portfolio, mode)
    before = hash_file(config.threshold.artifact)

    async def execute():
        with RunManager(config) as run:
            parts = assemble(config, run.logger)
            persist_thresholds(run, parts.thresholds)
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
                assert {r.final_trusted_tier for r in rows} == {t.value for t in portfolio}
                for row in rows:
                    assert row.fate in {"discard", "compress", "transmit"}
                    assert row.threshold_artifact_sha256 == before
                    assert row.input_bytes == 120032
                    assert row.fate_ms >= 0 and row.compression_cpu_ms == 0
                    assert row.additional_forward_passes == 0
                    assert (
                        row.output_bytes
                        == {"discard": 0, "compress": 30032, "transmit": 120032}[row.fate]
                    )
                    writer.append(row)
            finally:
                await engine.shutdown()
                writer.close()

    asyncio.run(execute())
    assert hash_file(config.threshold.artifact) == before


def test_artifact_compatibility_and_test_partition_freeze(tmp_path):
    config = configured(
        tmp_path, (ModelTier.TINY,), "sequential", common=True, fate="common_three_way"
    )
    path, digest = config.threshold.artifact, config.threshold.artifact_sha256
    load_fate_artifact(path, digest, config)
    raw = config.model_dump()
    raw["pipeline"]["partition"] = "test"
    with pytest.raises(ValueError, match="frozen promotion"):
        load_fate_artifact(path, digest, AppConfig.model_validate(raw))
    raw["pipeline"]["partition"] = "validation"
    raw["fate"]["compression_fraction"] = 0.5
    with pytest.raises(ValueError, match="incompatible"):
        load_fate_artifact(path, digest, AppConfig.model_validate(raw))
    with pytest.raises(ValueError, match="hash"):
        load_fate_artifact(path, "0" * 64, config)
    assert assemble(config).thresholds.resolve(ModelTier.TINY).artifact_sha256 == digest
    content = path.read_bytes()
    path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="hash"):
        load_fate_artifact(path, digest, config)


@pytest.mark.parametrize("failure", ["inference", "uq", "terminal"])
def test_failed_and_terminal_candidates_preserve_full_payload(tmp_path, failure):
    config = configured(tmp_path, (ModelTier.TINY,), "sequential")
    from edge_triage.contracts import EscalationDecision
    from edge_triage.runtime.reference import NoUncertaintyPolicy

    class FailingModel:
        def predict(self, item, tier):
            raise RuntimeError("injected model failure")

    class FailingUQ(NoUncertaintyPolicy):
        def evaluate(self, prediction):
            raise RuntimeError("injected UQ failure")

    class Unresolved:
        def decide(self, prediction, uncertainty):
            return EscalationDecision(
                candidate_id=prediction.candidate_id,
                from_tier=prediction.tier,
                escalate=False,
                conservative_fallback=True,
                reason_code="UNRESOLVED",
            )

    with RunManager(config) as run:
        parts = assemble(config, run.logger)
        if failure == "inference":
            parts.models[ModelTier.TINY] = FailingModel()
        elif failure == "uq":
            parts.uncertainty = FailingUQ()
        else:
            parts.escalation = Unresolved()
        writer = ParquetRecordWriter(run.directory / "candidates.parquet")
        pipeline = Pipeline(config, parts, run.logger, writer)
        assert pipeline.run(DatasetSource(config, run.manifest.run_id, 1)) == (
            (1, 0) if failure == "terminal" else (0, 1)
        )
        import pyarrow.parquet as pq

        row = pq.read_table(run.directory / "candidates.parquet").to_pylist()[0]
        assert row["fate"] == "transmit" and row["conservative_fallback"] == 1
        assert row["input_bytes"] == row["output_bytes"] == 120032
        assert row["fate_reason_code"].startswith("PRESERVE_")
