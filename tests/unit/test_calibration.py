import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from edge_triage.calibration.metrics import bootstrap, metrics, probabilities, recall_threshold
from edge_triage.calibration.profiling import measure, summarize
from edge_triage.calibration.temperature import (
    CalibratedModelAdapter,
    CalibrationArtifact,
    fit_temperature,
    load_calibration,
)
from edge_triage.calibration.workflow import validation_predictions
from edge_triage.config import AppConfig, EvaluationConfig, ProfilingConfig
from edge_triage.contracts import Candidate, ModelTier
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.models.cnn import ScalableRealBogusCNN, TierSpec
from edge_triage.models.registry import CheckpointManifest, RealModelAdapter
from edge_triage.runtime.interfaces import CandidateInput


def test_metrics_known_ranking_confusion_and_calibration():
    y = [0, 0, 1, 1]
    p = np.array([0.1, 0.4, 0.35, 0.8])
    z = np.log(p / (1 - p))
    result = metrics(y, z, bins=2)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["precision"] == 1
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["mcc"] == pytest.approx(1 / math.sqrt(3))
    assert result["roc_auc"] == 0.75
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["pr_auc_trapezoidal"] == pytest.approx(19 / 24)
    assert result["brier"] == pytest.approx(0.158125)
    assert result["ece"] == pytest.approx(0.0875)
    assert result["nll"] == pytest.approx(-np.mean(np.log([0.9, 0.6, 0.35, 0.8])))
    assert result["undefined"] == {}


def test_ties_extremes_and_undefined_are_explicit():
    tied = metrics([1, 0, 1, 0], [2, 2, 2, 2])
    assert tied["roc_auc"] == tied["average_precision"] == 0.5
    result = metrics([0, 0], [-1000, -1000])
    assert result["precision"] is result["recall"] is result["mcc"] is None
    assert result["roc_auc"] is result["average_precision"] is None
    assert result["undefined"]["roc_auc"] == "requires both classes"
    assert metrics([0, 1], [-1000, 1000])["nll"] == 0
    # Saturated sigmoid must not erase ranking information.
    assert metrics([0, 1], [1000, 1001])["roc_auc"] == 1


@pytest.mark.parametrize("y,z", [([], []), ([0], [float("nan")]), ([2], [0]), ([0, 1], [1])])
def test_invalid_metric_inputs(y, z):
    with pytest.raises(ValueError):
        metrics(y, z)


def test_threshold_ties_and_bootstrap_reproducibility():
    y, z = [1, 1, 1, 1, 0], [3, 2, 2, 1, 2]
    threshold = recall_threshold(y, z, 0.75)
    assert threshold == 2
    assert metrics(y, z, threshold_logit=threshold)["recall"] == 0.75
    kwargs = dict(temperature=1, threshold=threshold, bins=3, repeats=40, seed=7)
    assert bootstrap(y, z, **kwargs) == bootstrap(y, z, **kwargs)


def test_temperature_optimum_bounds_and_invariance():
    y, z = [1, 1, 1, 0, 0, 0, 0, 1], [2, 2, 2, 2, -2, -2, -2, -2]
    fitted = fit_temperature(y, z, partition="validation")
    assert fitted["temperature"] == pytest.approx(2 / math.log(3))
    assert metrics(y, z, temperature=fitted["temperature"])["nll"] < metrics(y, z)["nll"]
    assert (
        metrics(y, z, temperature=fitted["temperature"])["confusion_matrix"]
        == metrics(y, z)["confusion_matrix"]
    )
    assert (
        fit_temperature([0, 1], [-1, 1], partition="validation")["boundary"]
        == "minimum_temperature"
    )
    assert (
        fit_temperature([1, 0], [-1, 1], partition="validation")["boundary"]
        == "maximum_temperature"
    )
    for t in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            probabilities([1], t)


def test_no_test_access(monkeypatch):
    with pytest.raises(ValueError, match="validation-only"):
        fit_temperature([0, 1], [-1, 1], partition="test")
    config = AppConfig()
    config = config.model_copy(
        update={"pipeline": config.pipeline.model_copy(update={"partition": "test"})}
    )
    monkeypatch.setattr(
        "edge_triage.calibration.workflow.make_loader",
        lambda *a, **k: pytest.fail("loader must not be called"),
    )
    with pytest.raises(ValueError, match="forbids"):
        validation_predictions(config, None, ModelTier.TINY, "x", "run")
    with pytest.raises(ValidationError):
        EvaluationConfig(partition="test")
    with pytest.raises(ValidationError):
        ProfilingConfig(batch_sizes=(1, 2, 4, 4))


def artifact_fixture():
    return CalibrationArtifact(
        artifact_id="cal-1",
        checkpoint_id="test",
        tier=ModelTier.TINY,
        seed=1,
        checkpoint_manifest_sha256="manifest",
        split_sha256="split",
        preprocessing_sha256="prep",
        validation_rows_sha256="rows",
        prediction_sha256="pred",
        config_sha256="config",
        code_sha256="code",
        temperature=2,
        recommended_mode="calibrated",
        selection_criterion="test",
        fitting={},
        before={},
        after={},
    )


def test_calibration_roundtrip_hash_binding_and_adapter(tmp_path):
    artifact = artifact_fixture()
    path = tmp_path / "temperature.json"
    write_immutable(path, canonical(artifact.model_dump(mode="json")))
    manifest = CheckpointManifest(
        checkpoint_id="test",
        tier=ModelTier.TINY,
        architecture_version="scalable_cnn_v1",
        parameter_count=24649,
        dataset_hashes={},
        split_sha256="split",
        preprocessing_sha256="prep",
        seed=1,
        git_commit=None,
        best_epoch=1,
        selection_metric="validation_average_precision",
        file_hashes={},
    )
    assert load_calibration(path, hash_file(path), manifest, "manifest") == artifact
    with pytest.raises(ValueError, match="incompatible"):
        load_calibration(path, hash_file(path), manifest, "wrong")
    with pytest.raises(ValueError, match="hash"):
        load_calibration(path, "wrong", manifest, "manifest")
    model = ScalableRealBogusCNN(
        TierSpec(tier=ModelTier.TINY, base_filters=8, dense_width=64)
    ).eval()
    base = RealModelAdapter(model, "test")
    adapter = CalibratedModelAdapter(base, artifact)
    item = CandidateInput(
        Candidate(
            run_id="r", trace_id="t", candidate_id="c", dataset_index=0, arrival_monotonic_ns=0
        ),
        torch.randn(3, 30, 30),
        0,
        120000,
    )
    prediction = adapter.predict(item, ModelTier.TINY)
    assert prediction.p_real == pytest.approx(float(probabilities(prediction.logit)))
    assert prediction.calibrated_p_real == pytest.approx(float(probabilities(prediction.logit, 2)))
    assert prediction.calibration_artifact_id == "cal-1"
    assert (
        CalibratedModelAdapter(base, artifact, "uncalibrated")
        .predict(item, ModelTier.TINY)
        .calibrated_p_real
        is None
    )
    with pytest.raises(FileExistsError):
        write_immutable(path, b"{}")


def test_cuda_synchronization_order(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    samples = measure(lambda: calls.append("operation"), torch.device("cuda"), warmup=2, repeats=3)
    assert calls == ["operation", "operation"] + ["sync", "operation", "sync"] * 3
    assert len(samples) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA absent")
def test_real_cuda_timing_repeat_stability():
    device = torch.device("cuda")
    tensor = torch.randn(128, 128, device=device)
    rounds = [
        summarize(measure(lambda: tensor @ tensor, device, warmup=20, repeats=100))
        for _ in range(2)
    ]
    assert max(r.median_ms for r in rounds) / min(r.median_ms for r in rounds) < 5
    assert all(r.p99_ms >= r.median_ms > 0 for r in rounds)
