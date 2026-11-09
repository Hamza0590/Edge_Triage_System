"""Small real-model validation matrix shared by Module11 acceptance tests."""

import csv
import itertools
from pathlib import Path

import yaml

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import canonical
from edge_triage.health import hash_file

ROOT = Path(__file__).resolve().parents[2]


def matrix_fixture(directory, *, all_portfolios=False, repetitions=2):
    directory.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml")
    original = ROOT / "artifacts/triage/module10_v2/validation_trace.json"
    import json

    trace = json.loads(original.read_bytes())
    with base.split.manifest.open(encoding="utf-8", newline="") as stream:
        valid = [r for r in csv.DictReader(stream) if r["split"] == "validation"]
    selected = {
        r["candidate_id"]
        for label in ("0", "1")
        for r in [r for r in valid if r["label"] == label][:2]
    }
    trace["records"] = [r for r in trace["records"] if r["source_candidate_id"] in selected]
    for i, row in enumerate(trace["records"]):
        row["trace_position"] = i
    trace_path = directory / "trace.json"
    trace_path.write_bytes(canonical(trace))
    trace_path.with_suffix(".sha256").write_text(hash_file(trace_path), encoding="utf-8")
    portfolios = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]
    if not all_portfolios:
        portfolios = [(ModelTier.TINY,)]
    cases, groups = [], []
    for portfolio in portfolios:
        name = "+".join(t.value for t in portfolio)
        groups.append({"id": name, "changed_axes": ["execution_mode"]})
        for mode in ("sequential", "fixed_concurrency", "dynamic_concurrency"):
            case_id = f"{name}_{mode}"
            raw = base.model_dump()
            raw["models"]["tier_portfolio"] = portfolio
            raw["tier_selection"].update(policy="fixed", fixed_tier=portfolio[0])
            raw["admission"].update(
                execution_mode=mode, fixed_concurrency=2, concurrent_policy="fixed_memory"
            )
            raw["fate"]["policy"] = "transmit_all"
            raw["threshold"].update(policy="common", artifact=None, artifact_sha256=None)
            raw["workload"].update(trace=trace_path, count=4, profile="backlog")
            raw["paths"]["runs_dir"] = directory / "runs"
            config_path = directory / f"{case_id}.yaml"
            config_path.write_text(AppConfig.model_validate(raw).resolved_yaml(), encoding="utf-8")
            cases.append(
                {
                    "id": case_id,
                    "description": "Test-owned reduced real pipeline smoke",
                    "config": {"path": str(config_path), "sha256": hash_file(config_path)},
                    "comparison_group": name,
                }
            )
    value = {
        "schema_version": 1,
        "name": "test_matrix",
        "project_root": str(ROOT),
        "output_dir": str(directory / "study"),
        "repetitions": repetitions,
        "warmup_forwards_per_tier": 2,
        "bootstrap_repeats": 10,
        "cases": cases,
        "comparisons": groups,
    }
    matrix_path = directory / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return matrix_path
