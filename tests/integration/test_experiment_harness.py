import json

import pytest

from edge_triage.cli import main
from edge_triage.experiments import harness
from edge_triage.experiments.matrix import load_matrix
from tests.fixtures.experiments import matrix_fixture


def test_reduced_end_to_end_matrix_resume_and_plot(tmp_path, monkeypatch):
    path = matrix_fixture(tmp_path)
    assert main(["experiment", "validate-matrix", "--config", str(path)]) == 0
    report = harness.run_matrix(path)
    statuses = json.loads(report.read_bytes())["subruns"]
    assert len(statuses) == 6 and {s["status"] for s in statuses} == {"complete"}
    assert len({s["run"] for s in statuses}) == 6
    output = load_matrix(path).output_dir
    summary_path = harness.summarize_study(output)
    summary = json.loads(summary_path.read_bytes())
    assert summary["counts"] == {"complete": 6, "failed": 0, "pending": 0, "skipped": 0}
    for subrun in summary["subruns"]:
        metrics = subrun["metrics"]
        assert metrics["occurrences"] == 4 and metrics["unique_sources"] == 4
        assert metrics["science"] is None and metrics["science_scope"] == "systems_smoke_only"
        assert metrics["failures"] == 0 and metrics["total_forward_passes"] == 4
        assert metrics["warmup"]["total_forwards"] == 2
        assert metrics["costs"]["fine_components_ms"]["preprocessing_ms"] > 0
    index = harness.plot_summary(summary_path)
    assert len(json.loads(index.read_bytes())["plots"]) == 2
    assert harness.plot_summary(summary_path) == index

    def forbidden(*args, **kwargs):
        raise AssertionError("completed resume must not execute inference")

    monkeypatch.setattr(harness, "run_concurrent", forbidden)
    assert harness.run_matrix(path) == report
    assert harness.summarize_study(output) == summary_path
    assert main(["experiment", "summarize", "--runs", str(output)]) == 0
    marker = next((output / "subruns").glob("*/complete.json"))
    value = json.loads(marker.read_bytes())
    (marker.parent / value["metrics_file"]).write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        harness.run_matrix(path)
    assert not (output / "execution.lock").exists()


def test_failure_is_recorded_and_other_subruns_continue(tmp_path, monkeypatch):
    path = matrix_fixture(tmp_path)

    def failed(*args, **kwargs):
        raise RuntimeError("injected startup failure")

    monkeypatch.setattr(harness, "run_concurrent", failed)
    report = harness.run_matrix(path)
    assert all(r["status"] == "failed" for r in json.loads(report.read_bytes())["subruns"])
    output = load_matrix(path).output_dir
    assert len(list((output / "subruns").glob("*/failed_*.json"))) == 6
    assert not list((output / "subruns").glob("*/complete.json"))
    assert json.loads(harness.summarize_study(output).read_bytes())["counts"]["failed"] == 6
    # Retries preserve failed attempts and never reuse a completed runtime run ID.
    harness.run_matrix(path)
    assert len(list((output / "subruns").glob("*/failed_*.json"))) == 12


def test_execution_lock_refuses_concurrent_or_stale_owner(tmp_path):
    path = matrix_fixture(tmp_path)
    matrix, _ = harness.freeze_matrix(path)
    lock = matrix.output_dir / "execution.lock"
    lock.write_text('{"pid": 123}')
    with pytest.raises(FileExistsError):
        harness.run_matrix(path)
    assert lock.read_text() == '{"pid": 123}'


def test_component_backend_uses_same_matrix_and_resumes(tmp_path, monkeypatch):
    import numpy as np
    import yaml

    from edge_triage.config import load_config
    from edge_triage.experiments.components import inspect_component

    path = matrix_fixture(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["cases"] = raw["cases"][:2]
    for case, backend in zip(raw["cases"], ("microbatch", "batch1_threads")):
        case.update(component=backend, component_level=2)
    path.write_text(yaml.safe_dump(raw))
    report = harness.run_matrix(path)
    results = json.loads(report.read_bytes())["subruns"]
    assert len(results) == 4 and all(r["status"] == "complete" for r in results)
    from pathlib import Path

    outputs = [
        json.loads((Path(r["run"]) / "metrics/component.json").read_bytes()) for r in results
    ]
    for output in outputs[1:]:
        assert np.allclose(
            [r["logit"] for r in outputs[0]["rows"]],
            [r["logit"] for r in output["rows"]],
            atol=1e-5,
            rtol=1e-5,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("completed component must not rerun")

    monkeypatch.setattr(harness, "run_component", forbidden)
    assert harness.run_matrix(path) == report
    summary = json.loads(harness.summarize_study(load_matrix(path).output_dir).read_bytes())
    assert all(
        r["metrics"]["science"] is None and r["metrics"]["occurrences"] == 4
        for r in summary["subruns"]
    )
    run = Path(results[0]["run"])
    outputs[0]["rows"][0]["candidate_id"] = "wrong"
    (run / "metrics/component.json").write_text(json.dumps(outputs[0]))
    with pytest.raises(ValueError, match="identity/protocol"):
        inspect_component(load_config(run / "config.resolved.yaml"), run, "microbatch", 2, 2)
