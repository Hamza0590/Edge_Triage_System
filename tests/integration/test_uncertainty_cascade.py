"""Isolated diagnostic routing fixtures exercise actual dropout in every portfolio."""

import itertools
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.health import hash_file
from edge_triage.runtime.factory import assemble, run_smoke
from edge_triage.uncertainty.artifacts import TierUncertainty, UncertaintyArtifact

ROOT = Path(__file__).resolve().parents[2]
PORTFOLIOS = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]


@pytest.fixture
def uq_config(tmp_path):
    config = load_config(ROOT / "configs/experiments/module07_v1.yaml")
    raw = config.model_dump()
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    raw["calibration"]["probability_mode"] = "calibrated"
    raw["pipeline"]["mode"] = "reference"
    raw["tier_selection"]["policy"] = "fixed"
    config = AppConfig.model_validate(raw)
    parts = assemble(config)
    index = json.loads(config.calibration.artifact_index.read_text())
    artifact = UncertaintyArtifact(
        artifact_id="isolated-forced-routing-test",
        passes=5,
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        family_sha256=hash_file(config.models.checkpoint_index),
        calibration_index_sha256=hash_file(config.calibration.artifact_index),
        objective="Synthetic test-only bounds force all paths; never a fitted research artifact",
        tiers=tuple(
            TierUncertainty(
                tier=t,
                checkpoint_id=a.base.checkpoint_id,
                calibration_artifact_id=a.artifact.artifact_id,
                calibration_sha256=index["calibrations"][t.value]["sha256"],
                probability_mode="calibrated",
                temperature=a.artifact.temperature,
                lower=0,
                upper=1,
                std_threshold=0,
            )
            for t, a in parts.models.items()
        ),
    )
    path = tmp_path / "diagnostic_routing.json"
    path.write_text(artifact.model_dump_json())
    raw["uncertainty"].update(
        policy="selective_mc_dropout",
        passes=5,
        seed=123,
        artifact=path,
        artifact_sha256=hash_file(path),
    )
    raw["escalation"]["policy"] = "cascade"
    return AppConfig.model_validate(raw)


@pytest.mark.parametrize("portfolio", PORTFOLIOS)
def test_real_sequential_cascades_all_portfolios(uq_config, portfolio):
    raw = uq_config.model_dump()
    raw["models"]["tier_portfolio"] = portfolio
    raw["tier_selection"]["fixed_tier"] = portfolio[0]
    directory = run_smoke(AppConfig.model_validate(raw), limit=1)
    (row,) = pq.read_table(directory / "candidates.parquet").to_pylist()
    assert row["status"] == "completed"
    assert row["escalation_path"] == "->".join(t.value for t in portfolio)
    assert row["additional_forward_passes"] == 5 * len(portfolio)
    assert row["conservative_fallback"] == 1
    predictions = json.loads(row["prediction_chain_json"])
    uncertainty = json.loads(row["uncertainty_chain_json"])
    assert len(uncertainty) == len(predictions) == len(portfolio)
    assert all(u["duration_ms"] > 0 and u["diagnostic_seed"] is not None for u in uncertainty)
    assert row["final_trusted_p_real"] == predictions[-1]["calibrated_p_real"]
    events = [json.loads(x) for x in (directory / "events.jsonl").read_text().splitlines()]
    final = next(e["payload"] for e in events if e["event_type"] == "inference.candidate_final")
    assert final["prediction_chain"] == predictions and final["uncertainty_chain"] == uncertainty
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["seeds"]["uncertainty"] == 123


def test_runtime_rejects_pass_or_calibration_mismatch(uq_config):
    raw = uq_config.model_dump()
    raw["uncertainty"]["passes"] = 10
    with pytest.raises(ValueError, match="incompatible frozen UQ"):
        assemble(AppConfig.model_validate(raw))
    raw = uq_config.model_dump()
    raw["calibration"]["probability_mode"] = "uncalibrated"
    with pytest.raises(ValueError, match="UQ probability"):
        assemble(AppConfig.model_validate(raw))
