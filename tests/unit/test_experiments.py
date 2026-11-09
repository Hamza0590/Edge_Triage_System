import json

import pyarrow.parquet as pq
import pytest
import yaml

from edge_triage.config import AppConfig, load_config
from edge_triage.experiments import matrix as matrix_module
from edge_triage.experiments.harness import freeze_matrix, plot_summary, read_completed
from edge_triage.experiments.matrix import (
    ExperimentMatrix,
    enforce_comparison,
    input_inventory,
    load_matrix,
    validate_matrix,
)
from edge_triage.experiments.reporting import latency_partition, science_metrics, summarize_records
from edge_triage.health import hash_file
from edge_triage.runtime.records import CandidateRecord
from tests.fixtures.experiments import ROOT, matrix_fixture


@pytest.fixture
def matrix_path(tmp_path, monkeypatch):
    path = matrix_fixture(tmp_path)
    # Semantic matrix tests need not re-hash the 386MB raw file for every case.
    # Real inventory tamper coverage and integration below retain actual checks.
    monkeypatch.setattr(matrix_module, "input_inventory", lambda config: {})
    return path


def test_expansion_all_portfolios_three_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix_module, "input_inventory", lambda config: {})
    path = matrix_fixture(tmp_path, all_portfolios=True)
    result = validate_matrix(load_matrix(path))
    assert result["counts"] == {"cases": 21, "subruns": 42, "ready": 42, "skipped": 0}
    assert len({tuple(c["axes"]["portfolio"]) for c in result["cases"]}) == 7
    assert len({r["id"] for r in result["subruns"]}) == 42


def test_isolation_and_same_trace_split(matrix_path):
    config = load_config(load_matrix(matrix_path).cases[0].config.path)
    raw = config.model_dump()
    raw["admission"]["execution_mode"] = "fixed_concurrency"
    changed = AppConfig.model_validate(raw)
    enforce_comparison([config, changed], ("execution_mode",))
    for section, key, value in (
        ("calibration", "probability_mode", "uncalibrated"),
        ("workload", "trace", ROOT / "wrong_trace.json"),
        ("split", "manifest", ROOT / "wrong_split.csv"),
        ("training", "cpu_threads", 2),
    ):
        broken = changed.model_dump()
        broken[section][key] = value
        with pytest.raises(ValueError, match="undeclared"):
            enforce_comparison([config, AppConfig.model_validate(broken)], ("execution_mode",))


def test_unknown_axis_does_not_pretend_independence(matrix_path):
    config = load_config(load_matrix(matrix_path).cases[0].config.path)
    with pytest.raises(ValueError, match="unsupported independent"):
        enforce_comparison([config], ("invented_axis",))


def test_execution_comparison_rejects_hidden_auto_admission_change(matrix_path):
    config = load_config(load_matrix(matrix_path).cases[0].config.path)
    raw = config.model_dump()
    raw["admission"]["concurrent_policy"] = "auto"
    sequential = AppConfig.model_validate(raw)
    raw["admission"]["execution_mode"] = "dynamic_concurrency"
    dynamic = AppConfig.model_validate(raw)
    with pytest.raises(ValueError, match="effective admission"):
        enforce_comparison([sequential, dynamic], ("execution_mode",))
    enforce_comparison([sequential, dynamic], ("execution_mode", "admission_policy"))


def test_validation_partition_guard_before_execution(matrix_path):
    value = yaml.safe_load(matrix_path.read_text())
    value["partition"] = "test"
    with pytest.raises(ValueError):
        ExperimentMatrix.model_validate(value)


def test_unique_validation_refuses_reduced_or_repeated_sources(matrix_path):
    matrix = load_matrix(matrix_path).model_copy(update={"purpose": "unique_validation"})
    with pytest.raises(ValueError, match="every validation source"):
        validate_matrix(matrix)


def test_freeze_refuses_changed_matrix_and_source(matrix_path, monkeypatch):
    from edge_triage.experiments import harness

    freeze_matrix(matrix_path)
    freeze_matrix(matrix_path)
    monkeypatch.setattr(harness, "source_inventory", lambda root: {"modified.py": "f" * 64})
    with pytest.raises(FileExistsError):
        freeze_matrix(matrix_path)


def test_input_inventory_checks_transitive_checkpoint_pin(matrix_path, monkeypatch):
    from edge_triage.models import registry

    config = load_config(load_matrix(matrix_path).cases[0].config.path)
    original = registry.read_index(config.models.checkpoint_index)
    original["checkpoints"]["tiny"]["manifest_sha256"] = "0" * 64
    monkeypatch.setattr(matrix_module, "read_index", lambda path: original)
    with pytest.raises(ValueError, match="checksum"):
        input_inventory(config)


def test_unknown_keys_duplicate_cases_and_missing_skip_reason(matrix_path):
    value = yaml.safe_load(matrix_path.read_text())
    for change in (
        {"unexpected": True},
        {"cases": [*value["cases"], value["cases"][0]]},
        {"cases": [{"id": "skip", "description": "missing reason", "status": "skipped"}]},
    ):
        with pytest.raises(ValueError):
            ExperimentMatrix.model_validate({**value, **change})


@pytest.fixture
def saved_rows():
    collection = json.loads(
        (
            ROOT / "artifacts/triage/module10_v2/settings/core_tiny_sequential/collection.json"
        ).read_bytes()
    )
    from pathlib import Path

    return [
        CandidateRecord.model_validate(r)
        for r in pq.read_table(Path(collection["run"]) / "candidates.parquet").to_pylist()
    ]


def test_cost_partition_is_additive_even_when_fine_costs_overlap(saved_rows):
    for row in saved_rows:
        assert sum(latency_partition(row).values()) == pytest.approx(row.end_to_end_ms)


def test_target_failure_is_visible_without_refit(saved_rows):
    real = next(r for r in saved_rows if r.label == 1)
    broken = real.model_copy(update={"fate": "discard", "output_bytes": 0})
    result = science_metrics([broken], 0.95)
    assert result["survival_recall"] == 0 and result["false_real_discards"] == 1
    assert not result["equal_recall_eligible"] and not result["threshold_refitted"]
    result = science_metrics([real.model_copy(update={"fate": "transmit"})], 0.95)
    assert result["equal_recall_eligible"]


def test_repeated_sources_cannot_enter_scientific_metrics(saved_rows):
    rows = saved_rows[:2]
    sources = {r.candidate_id: "same_source" for r in rows}
    with pytest.raises(ValueError, match="replayed duplicates"):
        summarize_records(
            rows, sources, {}, [], unique_validation=True, target=0.95, bootstrap_repeats=10, seed=1
        )


def test_frozen_config_hash_tamper(matrix_path):
    matrix = load_matrix(matrix_path)
    config_path = matrix.cases[0].config.path
    config_path.write_text(config_path.read_text() + "\n# changed\n", encoding="utf-8")
    assert hash_file(config_path) != matrix.cases[0].config.sha256
    with pytest.raises(ValueError, match="config checksum"):
        validate_matrix(matrix)


def test_plot_rejects_wrong_summary_schema(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"schema_version": 999}')
    with pytest.raises(ValueError, match="schema"):
        plot_summary(path)


def test_saved_scientific_summary_and_all_plot_schemas(tmp_path, saved_rows):
    from pathlib import Path

    collection = json.loads(
        (
            ROOT / "artifacts/triage/module10_v2/settings/core_tiny_sequential/collection.json"
        ).read_bytes()
    )
    run = Path(collection["run"])
    runtime = json.loads((run / "metrics/concurrent_summary.json").read_bytes())
    # Historical Module10 records lack Module11 timers; placeholders are QA-only.
    runtime.update(stream_wall_ms=10000, warmup={"forwards_per_tier": 5})
    runtime["workload"]["preprocessing_ms"] = 1
    resources = [json.loads(line) for line in (run / "resources.jsonl").read_text().splitlines()]
    result = summarize_records(
        saved_rows,
        {r.candidate_id: r.candidate_id for r in saved_rows},
        runtime,
        resources,
        unique_validation=True,
        target=0.95,
        bootstrap_repeats=10,
        seed=1,
    )
    assert result["science"]["n"] == 643
    assert (
        result["science"]["classification_bootstrap"]["intervals"]["recall"]["low"]
        <= (result["science"]["classification_bootstrap"]["intervals"]["recall"]["high"])
    )
    # Saved validation data reused solely for plot/schema QA, not scientific comparisons.
    names = [
        "B1",
        "B2",
        "B3",
        "B4a",
        "B5",
        "B6",
        "B7",
        "FULL",
        "B5_common",
        "B5_arrival_20",
        "B5_arrival_100",
        "static_all",
        "portfolio_tiny",
        "portfolio_medium",
        "portfolio_large",
        "portfolio_tiny+medium",
        "portfolio_tiny+large",
        "portfolio_medium+large",
        "seed_20260924_tiny",
    ]
    value = dict(
        schema_version=1,
        purpose="unique_validation",
        subruns=[dict(case=name, status="complete", metrics=result) for name in names],
    )
    value["subruns"].append(
        dict(
            case="component_tiny_microbatch_1",
            status="complete",
            metrics=dict(science=None, component=dict(level=1), throughput_candidates_per_s=100),
        )
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(value))
    index = plot_summary(path)
    plots = json.loads(index.read_bytes())["plots"]
    assert len(plots) == 11
    for name, digest in plots.items():
        assert hash_file(index.parent / name) == digest
    assert plot_summary(path) == index


def test_completion_rejects_missing_inventory_before_file_access(tmp_path):
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study_sha256": "a" * 64,
                "subrun_id": "case__r000",
                "run_hashes": {},
                "run": "does-not-exist",
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete completion inventory"):
        read_completed(tmp_path, "a" * 64, "case__r000")
