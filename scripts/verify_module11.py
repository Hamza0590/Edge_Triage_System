"""Regenerate study metrics and compare saved component predictions without inference."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from edge_triage.config import load_config
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.components import inspect_component
from edge_triage.experiments.harness import (
    freeze_matrix,
    inspect_run,
    plot_summary,
    read_completed,
    summarize_study,
)
from edge_triage.health import hash_file
from edge_triage.triage.artifacts import load_fate_artifact
from edge_triage.triage.workflow import read_collection

ROOT = Path(__file__).resolve().parents[1]


def verify(out):
    matrix, resolved = freeze_matrix(out / "matrix.yaml")
    study = matrix.output_dir
    study_hash = hash_file(study / "matrix.resolved.json")
    baselines, components, reports = {}, [], []
    collection_candidates = 0
    for spec in json.loads((out / "declaration.json").read_bytes()):
        directory = out / "validation" / spec["id"]
        config = load_config(directory / "fit/config.yaml")
        rows, evidence = read_collection(directory / "collection.json", config)
        artifact = load_fate_artifact(
            config.threshold.artifact, config.threshold.artifact_sha256, config
        )
        if len(rows) != 643 or artifact.validation_survival_recall < 0.95:
            raise ValueError("validation fit is incomplete or infeasible")
        if evidence["summary"]["warmup"]["forwards_per_tier"] != 5:
            raise ValueError("validation warm-up mismatch")
        collection_candidates += len(rows)
    for subrun in resolved["subruns"]:
        if subrun["status"] == "skipped":
            reports.append(subrun)
            continue
        folder = study / "subruns" / subrun["id"]
        if not (folder / "complete.json").exists():
            reports.append({**subrun, "status": "incomplete"})
            continue
        saved = read_completed(folder, study_hash, subrun["id"])
        run = Path(saved["run"])
        config = load_config(study / "configs" / f"{subrun['case']}.yaml")
        print(f"VERIFY {subrun['id']}", flush=True)
        actual = (
            inspect_component(
                config,
                run,
                subrun["component"],
                subrun["component_level"],
                matrix.warmup_forwards_per_tier,
            )
            if subrun["component"]
            else inspect_run(config, run, matrix)
        )
        expected = json.loads((folder / saved["metrics_file"]).read_bytes())
        for key in ("total_invocation_wall_ms", "total_invocation_scope"):
            expected.pop(key)
        if actual != expected:
            raise ValueError(f"saved metrics do not regenerate: {subrun['id']}")
        if subrun["case"] in {"B1", "B2", "B3"}:
            baselines.setdefault(subrun["case"], []).append(
                {
                    r["candidate_id"]: r["final_logit"]
                    for r in pq.read_table(run / "candidates.parquet").to_pylist()
                }
            )
        if subrun["component"]:
            components.append((subrun, json.loads((run / "metrics/component.json").read_bytes())))
        reports.append(
            {
                "id": subrun["id"],
                "status": "verified",
                "kind": saved["kind"],
                "occurrences": actual["occurrences"],
                "failures": actual["failures"],
                "equal_recall_eligible": actual["science"]["equal_recall_eligible"]
                if actual["science"]
                else None,
            }
        )
    equivalence = []
    for subrun, output in components:
        tier = subrun["case"].split("_")[1]
        key = {"tiny": "B2", "medium": "B3", "large": "B1"}[tier]
        predicted = {r["candidate_id"]: r["logit"] for r in output["rows"]}
        for reference in baselines[key]:
            if predicted.keys() != reference.keys():
                raise ValueError("component/pipeline candidate sets differ")
            a = np.array([predicted[k] for k in sorted(reference)])
            b = np.array([reference[k] for k in sorted(reference)])
            if not np.allclose(a, b, atol=1e-5, rtol=1e-5) or not np.array_equal(a >= 0, b >= 0):
                raise ValueError("component predictions differ from matched batch1 pipeline")
        equivalence.append(
            {
                "subrun": subrun["id"],
                "references": len(baselines[key]),
                "max_absolute_logit_difference": float(np.max(np.abs(a - b))),
            }
        )
    summary_path = summarize_study(study)
    summary = json.loads(summary_path.read_bytes())
    plots = plot_summary(summary_path)
    output = {
        "schema_version": 1,
        "matrix_sha256": study_hash,
        "summary": str(summary_path),
        "summary_sha256": hash_file(summary_path),
        "plots": str(plots),
        "plots_sha256": hash_file(plots),
        "validation_candidates": collection_candidates,
        "counts": summary["counts"],
        "subruns": reports,
        "component_equivalence": equivalence,
        "verification_inference": False,
        "scientific_test_evaluated": True,
        "thresholds_refitted": False,
    }
    analysis = out / "analysis"
    write_immutable(analysis / "verification.json", canonical(output))
    fields = [
        "case",
        "repetition",
        "kind",
        "n",
        "failures",
        "survival_recall",
        "false_real_discards",
        "equal_recall_eligible",
        "byte_reduction",
        "output_bytes",
        "throughput_per_s",
        "p95_end_to_end_ms",
        "stream_wall_ms",
        "forward_passes",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in summary["subruns"]:
        if row["status"] != "complete":
            continue
        metrics = row["metrics"]
        science = metrics["science"] or {}
        writer.writerow(
            {
                "case": row["case"],
                "repetition": row["run_repetition"],
                "kind": "component" if metrics.get("component") else "pipeline",
                "n": metrics["occurrences"],
                "failures": metrics["failures"],
                **{k: science.get(k) for k in fields[5:10]},
                "throughput_per_s": metrics["throughput_candidates_per_s"],
                "p95_end_to_end_ms": metrics["latency_ms"]["end_to_end_ms"]["p95"],
                "stream_wall_ms": metrics["stream_wall_ms"],
                "forward_passes": metrics.get("total_forward_passes"),
            }
        )
    write_immutable(analysis / "results.csv", stream.getvalue().encode())
    print(json.dumps({"counts": summary["counts"], "plots": str(plots)}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/module11_v3")
    args = parser.parse_args()
    verify(args.output.resolve())
