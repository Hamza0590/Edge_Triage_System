"""Declare a reduced 7-portfolio/3-mode validation systems smoke, never a test study."""

import argparse
import csv
import itertools
from pathlib import Path

import yaml

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.matrix import ExperimentMatrix
from edge_triage.experiments.workloads import TraceRecord
from edge_triage.health import hash_file

ROOT = Path(__file__).resolve().parents[1]


def prepare(directory):
    base = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml")
    with base.split.manifest.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "validation"]
    # Balanced engineering smoke; explicitly not a scientific or threshold-fitting sample.
    chosen = sorted(
        [r for label in ("0", "1") for r in [r for r in rows if r["label"] == label][:3]],
        key=lambda r: r["candidate_id"],
    )
    trace_path = directory / "inputs/smoke_trace.json"
    records = [
        TraceRecord(
            trace_position=i,
            candidate_id=row["candidate_id"],
            source_candidate_id=row["candidate_id"],
            dataset_index=int(row["index_no"]),
            relative_arrival_seconds=0,
            workload_phase="backlog",
            seed=20260924,
            repetition=0,
        ).model_dump(mode="json")
        for i, row in enumerate(chosen)
    ]
    write_immutable(
        trace_path,
        canonical(
            {
                "schema_version": 1,
                "partition": "validation",
                "split_sha256": hash_file(base.split.manifest),
                "records": records,
                "purpose": "MODULE11_REDUCED_SYSTEMS_SMOKE_NOT_SCIENTIFIC",
            }
        ),
    )
    write_immutable(trace_path.with_suffix(".sha256"), (hash_file(trace_path) + "\n").encode())
    cases, comparisons = [], []
    for n in (1, 2, 3):
        for portfolio in itertools.combinations(ModelTier, n):
            name = "+".join(t.value for t in portfolio)
            comparisons.append({"id": name, "changed_axes": ["execution_mode", "admission_policy"]})
            for mode in ("sequential", "fixed_concurrency", "dynamic_concurrency"):
                case_id = f"{name}_{mode}"
                raw = base.model_dump()
                raw["models"]["tier_portfolio"] = portfolio
                raw["tier_selection"].update(
                    policy="fixed" if n == 1 else "resource_queue_aware", fixed_tier=portfolio[0]
                )
                raw["admission"].update(execution_mode=mode, fixed_concurrency=2)
                raw["workload"].update(trace=trace_path.resolve(), profile="backlog", count=6)
                raw["fate"]["policy"] = "transmit_all"
                raw["threshold"].update(policy="common", artifact=None, artifact_sha256=None)
                raw["experiment"]["name"] = f"module11_smoke_{case_id}"
                raw["paths"]["runs_dir"] = (directory / "runtime_runs").resolve()
                config = AppConfig.model_validate(raw)
                path = directory / "inputs" / f"{case_id}.yaml"
                write_immutable(path, config.resolved_yaml().encode())
                cases.append(
                    {
                        "id": case_id,
                        "description": f"Reduced systems smoke: {name}, {mode}",
                        "config": {"path": str(path.resolve()), "sha256": hash_file(path)},
                        "comparison_group": name,
                    }
                )
    pending = {
        "B4a": "Tiny to Medium cascade",
        "B4b": "Tiny to Large cascade",
        "B4c": "Medium to Large cascade",
        "B4d": "Tiny to Medium to Large cascade",
        "FULL": "Selective UQ dynamic system (negative ablation; noUQ remains default)",
        "threshold_common_vs_per_tier": "Matched separately validated threshold policies",
        "queue_resource_only": "Independent queue-awareness control",
        "independent_admission": "Fixed executor with independently selected admission",
        "microbatch_vs_batch1": "Matched component microbatch comparison",
        "model_seed_diagnostics": "All three frozen model seeds",
    }
    for name, description in pending.items():
        cases.append(
            {
                "id": name,
                "description": description,
                "status": "skipped",
                "reason": "Outside reduced smoke; full matrix/protocol implementation pending",
            }
        )
    matrix = ExperimentMatrix.model_validate(
        {
            "name": "module11_smoke_v1",
            "project_root": ROOT,
            "output_dir": (directory / "study").resolve(),
            "purpose": "systems_smoke",
            "repetitions": 2,
            "warmup_forwards_per_tier": 5,
            "bootstrap_repeats": 100,
            "cases": cases,
            "comparisons": comparisons,
        }
    )
    path = directory / "matrix.yaml"
    write_immutable(path, yaml.safe_dump(matrix.model_dump(mode="json"), sort_keys=False).encode())
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/experiments/module11_smoke_v1"
    )
    prepare(parser.parse_args().output.resolve())
