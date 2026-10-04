"""Validate frozen Module08 measurements and replay sequential adaptive cascades."""

import argparse
import contextlib
import io
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from preflight_module07 import verify as preflight

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import CandidateEvent, ModelTier, ResourceSnapshot
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.data.loaders import MeerlichtDataset
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import dataset_hashes, hash_file
from edge_triage.runtime.records import CandidateRecord
from edge_triage.runtime.streaming import run_adaptive_sequential
from edge_triage.uncertainty.analysis import cascade_analysis, fit_interval, fit_std
from edge_triage.uncertainty.artifacts import load_uncertainty

ROOT = Path(__file__).resolve().parents[1]


def verify_measurements(index_path):
    index = json.loads(index_path.read_text())
    directory = index_path.parent
    for name, digest in index["files"].items():
        assert hash_file(directory / name) == digest, name
    objective = json.loads((directory / "objective.json").read_text())
    assert index["partition"] == "validation" and index["test_evaluated"] is False
    for name in ("objective", "protocol"):
        stem = "analysis_protocol" if name == "protocol" else name
        path = ROOT / f"configs/experiments/module08_{stem}_v1.json"
        assert hash_file(path) == index[f"{name}_sha256"]
        assert json.loads(path.read_text()) == json.loads((directory / f"{name}.json").read_text())
    config = load_config(ROOT / "configs/experiments/module07_v1.yaml")
    dataset = MeerlichtDataset(config, "validation", run_id="verify-module08")
    expected = {r["candidate_id"]: r["label"] for r in dataset.rows}
    drift = {}
    for n in (5, 10, 20):
        entry = index["artifacts"][str(n)]
        artifact = load_uncertainty(Path(entry["path"]), entry["sha256"])
        rows = json.loads((directory / f"samples_{n}.json").read_text())
        analysis = json.loads((directory / f"analysis_{n}.json").read_text())
        for spec in artifact.tiers:
            records = rows[spec.tier.value]
            assert len(records) == len(expected)
            assert {r["candidate_id"]: r["label"] for r in records} == expected
            p = [r["p"] for r in records]
            assert (spec.lower, spec.upper) == fit_interval(
                p, objective["ambiguous_fraction"], partition="validation"
            )
            assert spec.std_threshold == fit_std(
                [r["std"] for r in records if r["ambiguous"]],
                objective["std_quantile"],
                partition="validation",
            )
            for r in records:
                assert len(r["samples"]) == n and r["passes"] == n
                assert abs(float(np.mean(r["samples"])) - r["mean"]) < 1e-12
                assert abs(float(np.std(r["samples"])) - r["std"]) < 1e-12
                assert r["ambiguous"] == (spec.lower <= r["p"] <= spec.upper)
                assert r["unresolved"] == (
                    r["std"] >= spec.std_threshold or (r["mean"] >= 0.5) != (r["p"] >= 0.5)
                )
                assert r["deterministic_ms"] > 0 and r["mc_ms"] > 0
            drift[spec.tier.value] = {
                "max_absolute_batch1_vs_frozen_logit": max(
                    abs(r["logit"] - r["observed_batch1_logit"]) for r in records
                ),
                "frozen_threshold_crossings": sum(
                    (r["logit"] >= r["frozen_threshold_logit"])
                    != (r["observed_batch1_logit"] >= r["frozen_threshold_logit"])
                    for r in records
                ),
            }
        for report in analysis["cascades"]:
            assert report == cascade_analysis(rows, tuple(report["portfolio"]), mode=report["mode"])
        g = index["gate_c"]["budgets"][str(n)]
        assert g["pass"] == all(g["checks"].values())
    assert index["gate_c"]["pass"] == any(g["pass"] for g in index["gate_c"]["budgets"].values())
    return index, drift


def configs(index):
    base = ROOT / "configs/experiments/module07_v1.yaml"
    raw = yaml.safe_load(base.read_text())
    raw["experiment"].update(name="module08_selective_negative_v1")
    raw["experiment"]["provenance_ids"] += [
        "mc-dropout",
        "temperature-scaling",
        "human-label-free-uq",
    ]
    entry = index["artifacts"][str(index["gate_c"]["selected_passes"] or 5)]
    raw["calibration"]["probability_mode"] = "calibrated"
    raw["uncertainty"] = {
        "policy": "selective_mc_dropout",
        "passes": index["gate_c"]["selected_passes"] or 5,
        "seed": 20260924,
        "artifact": entry["path"],
        "artifact_sha256": entry["sha256"],
    }
    raw["escalation"] = {"policy": "cascade", "max_escalations": 2}
    diagnostic = ROOT / "configs/experiments/module08_selective_v1.yaml"
    write_immutable(diagnostic, yaml.safe_dump(raw, sort_keys=False).encode())
    raw["experiment"]["name"] = "module08_proposed_v1"
    if not index["gate_c"]["pass"]:
        raw["uncertainty"] = {"policy": "none", "seed": None}
        raw["escalation"] = {"policy": "none"}
        raw["calibration"]["probability_mode"] = "recommended"
    proposed = ROOT / "configs/experiments/module08_proposed_v1.yaml"
    write_immutable(proposed, yaml.safe_dump(raw, sort_keys=False).encode())
    return load_config(diagnostic), load_config(proposed)


def verify(index_path, output):
    evidence = preflight()
    index, drift = verify_measurements(index_path)
    config, proposed = configs(index)
    suite = json.loads((ROOT / "artifacts/traces/suite_a515d472c9d5ef05.json").read_text())
    backlog = Path(
        next(
            e["path"]
            for e in suite["traces"]
            if e["profile"] == "backlog" and e["process"] == "periodic"
        )
    )
    portfolios = [p for n in (1, 2, 3) for p in itertools.combinations(ModelTier, n)]
    jobs = [(config, p, backlog) for p in portfolios]
    jobs += [
        (config, tuple(ModelTier), config.workload.trace),
        (proposed, tuple(ModelTier), proposed.workload.trace),
    ]
    runs = []
    for base, portfolio, trace in jobs:
        raw = base.model_dump()
        raw["models"]["tier_portfolio"] = portfolio
        raw["workload"]["trace"] = trace
        current = AppConfig.model_validate(raw)
        with contextlib.redirect_stderr(io.StringIO()):
            directory = run_adaptive_sequential(current)
        summary_path = directory / "metrics/adaptive_sequential_summary.json"
        summary = json.loads(summary_path.read_text())
        rows = pq.read_table(directory / "candidates.parquet").to_pylist()
        trace_rows = load_trace(trace, hash_file(config.split.manifest))
        assert summary["completed"] == len(trace_rows) and summary["failed"] == 0
        assert {r["candidate_id"] for r in rows} == {r.candidate_id for r in trace_rows}
        assert len(rows) == len(trace_rows)
        assert (
            sum(
                summary["queue"][k]
                for k in ("rejected", "dropped", "cancelled", "active", "length")
            )
            == 0
        )
        for name, digest in summary["artifacts"].items():
            assert hash_file(directory / name) == digest
        resources = [
            ResourceSnapshot.model_validate_json(x)
            for x in (directory / "resources.jsonl").read_text().splitlines()
        ]
        snapshot_ids = {r.snapshot_id for r in resources}
        events = [
            CandidateEvent.model_validate_json(x)
            for x in (directory / "events.jsonl").read_text().splitlines()
        ]
        for row in rows:
            CandidateRecord.model_validate(row)
            assert row["concurrency"] == 1
            predictions = json.loads(row["prediction_chain_json"])
            uq = json.loads(row["uncertainty_chain_json"])
            escalations = json.loads(row["escalation_chain_json"])
            path = [p["tier"] for p in predictions]
            order = [t.value for t in portfolio]
            assert all(t in order for t in path)
            assert [order.index(t) for t in path] == sorted(set(order.index(t) for t in path))
            assert len(predictions) == len(uq) == len(escalations) <= 3
            assert row["selected_tiers"] == ",".join(path)
            assert row["additional_forward_passes"] == sum(
                u["additional_forward_passes"] for u in uq
            )
            assert row["uq_ms"] >= sum(u["duration_ms"] for u in uq)
            assert row["final_trusted_tier"] == path[-1]
            last = predictions[-1]
            assert row["final_trusted_p_real"] == (
                last["calibrated_p_real"]
                if last["calibrated_p_real"] is not None
                else last["p_real"]
            )
            assert row["conservative_fallback"] == int(escalations[-1]["conservative_fallback"])
            for ref in json.loads(row["resource_snapshot_json"]):
                assert ref["snapshot_id"] in snapshot_ids
            correlated = [e for e in events if e.candidate_id == row["candidate_id"]]
            assert all(
                e.trace_id == row["trace_id"] and e.run_id == row["run_id"] for e in correlated
            )
            for chain, kind in [
                (predictions, "inference.predicted"),
                (uq, "uncertainty.evaluated"),
                (escalations, "uncertainty.escalation"),
            ]:
                assert chain == [
                    e.model_dump(mode="json")["payload"] for e in correlated if e.event_type == kind
                ]
            final = next(
                e.model_dump(mode="json")["payload"]
                for e in correlated
                if e.event_type == "inference.candidate_final"
            )
            assert final["prediction_chain"] == predictions and final["uncertainty_chain"] == uq
            assert "_truncated" not in json.dumps(final) and "TRUNCATED" not in json.dumps(final)
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        result = {
            "run": str(directory),
            "portfolio": [t.value for t in portfolio],
            "policy": current.uncertainty.policy,
            "summary": summary,
            "summary_sha256": hash_file(summary_path),
            "manifest_sha256": hash_file(directory / "manifest.json"),
            "paths": dict(Counter(r["escalation_path"] for r in rows)),
            "total_additional_forward_passes": sum(r["additional_forward_passes"] for r in rows),
            "total_uq_ms": sum(r["uq_ms"] for r in rows),
            "total_escalation_decision_ms": sum(r["escalation_decision_ms"] for r in rows),
        }
        runs.append(result)
        print(f"{directory.name}: {len(rows)} completed; {result['paths']}", flush=True)
    write_immutable(
        output,
        canonical(
            {
                "verified": True,
                "test_evaluated": False,
                "classification_metrics": False,
                "preflight": evidence,
                "benchmark_index": str(index_path),
                "benchmark_index_sha256": hash_file(index_path),
                "batch1_drift": drift,
                "runs": runs,
                "completed": sum(r["summary"]["completed"] for r in runs),
                "dataset_hashes": [a.model_dump(mode="json") for a in dataset_hashes(config)],
                "config_hashes": {
                    "selective": config.config_hash,
                    "proposed": proposed.config_hash,
                },
            }
        ),
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.index, args.output)
