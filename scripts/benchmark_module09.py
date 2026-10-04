"""Immutable Module09 full-pipeline curves and component comparisons. No fitting/test data."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
import pyarrow.parquet as pq
import torch
from preflight_module09 import verify

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import CandidateEvent, ModelTier, ResourceSnapshot
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import hash_file
from edge_triage.runtime.streaming import TraceSource, run_concurrent
from edge_triage.scheduling.contention import ContentionProfile
from edge_triage.scheduling.profiles import scheduler_profiles

ROOT = Path(__file__).resolve().parents[1]


def quantiles(values):
    return {
        name: float(np.percentile(values, q)) for name, q in (("p50", 50), ("p95", 95), ("p99", 99))
    }


def inspect_run(directory, trace):
    rows = pq.read_table(directory / "candidates.parquet").to_pylist()
    by_id = {r["candidate_id"]: r for r in rows}
    assert set(by_id) == {r.candidate_id for r in trace} and len(rows) == len(trace)
    summary = json.loads((directory / "metrics/concurrent_summary.json").read_bytes())
    events = []
    finals = {}
    for line in (directory / "events.jsonl").read_text().splitlines():
        event = CandidateEvent.model_validate_json(line)
        events.append(event)
        if event.event_type == "inference.candidate_final":
            assert event.candidate_id not in finals
            finals[event.candidate_id] = event
    assert set(finals) == set(by_id)
    resources = [
        ResourceSnapshot.model_validate_json(line)
        for line in (directory / "resources.jsonl").read_text().splitlines()
    ]
    snapshot_ids = {r.snapshot_id for r in resources}
    for row in rows:
        assert row["trace_id"] == finals[row["candidate_id"]].trace_id
        for ref in json.loads(row["resource_snapshot_json"]):
            assert ref.get("snapshot_id") in snapshot_ids
        executions = json.loads(row["execution_chain_json"])
        for execution in executions:
            assert (
                execution["candidate_id"] == row["candidate_id"]
                and execution["trace_id"] == row["trace_id"]
            )
        assert abs(sum(e["inference_ms"] for e in executions) - row["inference_ms"]) < 1e-6
        assert abs(sum(e["uq_ms"] for e in executions) - row["uq_ms"]) < 1e-6
    elapsed = (max(r["final_ns"] for r in rows) - min(r["arrival_ns"] for r in rows)) / 1e9
    result = {
        "run": str(directory),
        "count": len(rows),
        "failed": sum(r["status"] != "completed" for r in rows),
        "throughput_per_s": len(rows) / elapsed,
        "elapsed_s": elapsed,
        "mean_inference_ms": float(np.mean([r["inference_ms"] for r in rows])),
        "inference_ms": quantiles([r["inference_ms"] for r in rows]),
        "end_to_end_ms": quantiles([r["end_to_end_ms"] for r in rows]),
        "queue_ms": quantiles([r["queue_ms"] for r in rows]),
        "admission_ms_total": sum(r["admission_ms"] for r in rows),
        "executor_completion_ms_total": sum(r["executor_ms"] for r in rows),
        "scheduler": summary["scheduler"],
        "executor": summary["executor"],
        "monitor": summary["monitor"],
        "workload": summary["workload"],
        "out_of_order": [r["candidate_id"] for r in rows] != [r.candidate_id for r in trace],
        "resource_maxima": {
            name: max(
                (getattr(r, name) for r in resources if getattr(r, name) is not None), default=None
            )
            for name in (
                "cpu_percent",
                "process_cpu_percent",
                "process_rss_bytes",
                "gpu_utilization_percent",
                "gpu_allocated_bytes",
                "gpu_reserved_bytes",
            )
        },
        "hashes": {
            name: hash_file(directory / name)
            for name in (
                "manifest.json",
                "events.jsonl",
                "resources.jsonl",
                "candidates.parquet",
                "metrics/concurrent_summary.json",
            )
        },
    }
    return result, by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["pipeline", "components", "finish"], required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/experiments/module09_v1.yaml")
    objective_path = ROOT / "configs/experiments/module09_objective_v1.json"
    objective = json.loads(objective_path.read_bytes())
    suite = json.loads((ROOT / "artifacts/traces/suite_a515d472c9d5ef05.json").read_bytes())
    traces = {
        entry["profile"]: Path(entry["path"])
        for entry in suite["traces"]
        if entry["process"] == "periodic"
    }
    trace = load_trace(traces["backlog"], hash_file(config.split.manifest))
    if args.stage == "pipeline":
        write_immutable(out / "preflight.json", canonical(verify()))
        write_immutable(out / "objective.json", objective_path.read_bytes())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        profiles = scheduler_profiles(config)
        for n in (1, 2, 3):
            for portfolio in itertools.combinations(ModelTier, n):
                key = "+".join(t.value for t in portfolio)
                reference = {}
                unsafe = False
                for level in objective["fixed_levels"]:
                    path = out / f"pipeline_{key}_{level}.json"
                    if path.exists():
                        saved = json.loads(path.read_bytes())
                        if level == 1:
                            reference = {
                                r["candidate_id"]: r
                                for r in pq.read_table(
                                    Path(saved["rounds"][0]["run"]) / "candidates.parquet"
                                ).to_pylist()
                            }
                        continue
                    needed = (
                        level * max(profiles[t].memory_bytes for t in portfolio)
                        + config.scheduler.constraints.memory_margin_bytes
                    )
                    unsafe = unsafe or needed > psutil.virtual_memory().available
                    result = {
                        "portfolio": key,
                        "level": level,
                        "rounds": [],
                        "skipped": unsafe,
                        "skip_reason": "memory precheck or lower-level failure" if unsafe else None,
                    }
                    if not unsafe:
                        for repeat in range(objective["rounds"]):
                            raw = config.model_dump()
                            raw["models"]["tier_portfolio"] = portfolio
                            raw["tier_selection"].update(
                                policy="round_robin", fixed_tier=portfolio[0]
                            )
                            raw["admission"]["fixed_concurrency"] = level
                            raw["workload"]["trace"] = traces["backlog"]
                            with (
                                open(os.devnull, "w") as console,
                                contextlib.redirect_stderr(console),
                            ):
                                directory = run_concurrent(AppConfig.model_validate(raw))
                            row, predictions = inspect_run(directory, trace)
                            if not reference:
                                reference = predictions
                            for identity, actual in predictions.items():
                                expected = reference[identity]
                                assert actual["selected_tiers"] == expected["selected_tiers"]
                                if actual["status"] == expected["status"] == "completed":
                                    assert (
                                        abs(actual["final_logit"] - expected["final_logit"]) < 1e-5
                                    )
                                    assert (
                                        abs(actual["final_p_real"] - expected["final_p_real"])
                                        < 1e-6
                                    )
                                    assert (actual["final_p_real"] >= 0.5) == (
                                        expected["final_p_real"] >= 0.5
                                    )
                            row["round"] = repeat
                            result["rounds"].append(row)
                            print(
                                f"pipeline {key} C={level} round={repeat} "
                                f"{row['throughput_per_s']:.2f}/s failed={row['failed']}",
                                flush=True,
                            )
                            if row["failed"]:
                                unsafe = True
                                break
                    write_immutable(path, canonical(result))
    elif args.stage == "components":
        # Transform frozen trace inputs once; pass only integer indices to processes.
        import threading

        source = TraceSource(config, "component", trace, threading.Event())
        images_path = out / "component_images.npy"
        if not images_path.exists():
            arrays = [source.dataset[source.positions[r.dataset_index]][1].numpy() for r in trace]
            with images_path.open("xb") as stream:
                np.save(stream, np.stack(arrays))
        for workload in ("backlog", "moderate"):
            rows = load_trace(traces[workload], hash_file(config.split.manifest))
            write_immutable(
                out / f"arrivals_{workload}.json",
                canonical([r.relative_arrival_seconds for r in rows]),
            )
        profiles = scheduler_profiles(config)
        for tier in ModelTier:
            for backend in ("threads", "processes", "microbatch"):
                unsafe = False
                for level in (1, 2, 4, 8):
                    needed = (
                        profiles[tier].memory_bytes * (level if backend == "processes" else 1)
                        + config.scheduler.constraints.memory_margin_bytes
                    )
                    unsafe = unsafe or needed > psutil.virtual_memory().available
                    for workload in (
                        ("backlog", "moderate") if backend == "microbatch" else ("backlog",)
                    ):
                        output = out / f"component_{tier.value}_{backend}_{level}_{workload}.json"
                        if output.exists():
                            continue
                        if unsafe:
                            write_immutable(
                                output,
                                canonical(
                                    {
                                        "skipped": True,
                                        "reason": "memory precheck",
                                        "needed_bytes": needed,
                                    }
                                ),
                            )
                            continue
                        command = [
                            sys.executable,
                            str(ROOT / "scripts/module09_components.py"),
                            "--config",
                            str(ROOT / "configs/experiments/module09_v1.yaml"),
                            "--images",
                            str(images_path),
                            "--tier",
                            tier.value,
                            "--backend",
                            backend,
                            "--level",
                            str(level),
                            "--output",
                            str(output),
                            "--arrivals",
                            str(out / f"arrivals_{workload}.json"),
                        ]
                        done = subprocess.run(command, capture_output=True, text=True, timeout=240)
                        if done.returncode:
                            write_immutable(
                                out / f"{output.stem}_failure.json",
                                canonical(
                                    {
                                        "command": command,
                                        "returncode": done.returncode,
                                        "stderr": done.stderr,
                                    }
                                ),
                            )
                            raise RuntimeError(done.stderr[-3000:])
                        print(done.stdout.strip(), flush=True)
    else:
        results = [json.loads(p.read_bytes()) for p in sorted(out.glob("pipeline_*.json"))]
        assert len(results) == 28
        caps, analyses = {}, []
        criteria = objective["gate_d"]
        for key in sorted({r["portfolio"] for r in results}):
            group = [r for r in results if r["portfolio"] == key]
            baseline = next(r for r in group if r["level"] == 1)["rounds"]
            choices = []
            for row in group:
                comparisons = []
                for actual, isolated in zip(row["rounds"], baseline, strict=False):
                    comparisons.append(
                        {
                            "throughput_gain": actual["throughput_per_s"]
                            / isolated["throughput_per_s"],
                            "latency_degradation": actual["mean_inference_ms"]
                            / isolated["mean_inference_ms"],
                            "e2e_p95_ratio": actual["end_to_end_ms"]["p95"]
                            / isolated["end_to_end_ms"]["p95"],
                        }
                    )
                passed = (
                    row["level"] > 1
                    and not row["skipped"]
                    and len(comparisons) == 3
                    and all(r["failed"] == 0 for r in row["rounds"])
                    and all(
                        r["throughput_gain"] >= criteria["minimum_throughput_gain_each_round"]
                        and r["latency_degradation"]
                        <= criteria["maximum_mean_inference_degradation_each_round"]
                        and r["e2e_p95_ratio"] <= criteria["maximum_e2e_p95_ratio_each_round"]
                        for r in comparisons
                    )
                )
                analyses.append(
                    {"portfolio": key, "level": row["level"], "rounds": comparisons, "pass": passed}
                )
                if passed:
                    choices.append(
                        (
                            float(np.median([r["throughput_per_s"] for r in row["rounds"]])),
                            -row["level"],
                        )
                    )
            caps[key] = -max(choices)[1] if choices else 1
        benchmark = {
            "objective_sha256": hash_file(objective_path),
            "analyses": analyses,
            "caps": caps,
            "gate_d_pass": any(c > 1 for c in caps.values()),
            "scope": "CPU frozen backlog, three repeats; conservative unknown-condition cap one",
            "files": {
                p.name: hash_file(p)
                for p in sorted(out.glob("*.json"))
                if p.name not in ("benchmark.json", "contention.json")
            },
        }
        write_immutable(out / "benchmark.json", canonical(benchmark))
        profile = ContentionProfile(
            hardware_id=config.scheduler.hardware_id,
            device="cpu",
            family_sha256=hash_file(config.models.checkpoint_index),
            profiles_sha256=config.scheduler.profiles_sha256,
            benchmark_sha256=hash_file(out / "benchmark.json"),
            criterion=json.dumps(criteria),
            caps=caps,
        )
        write_immutable(out / "contention.json", canonical(profile.model_dump(mode="json")))
        print(
            json.dumps(
                {
                    "gate_d_pass": benchmark["gate_d_pass"],
                    "caps": caps,
                    "profile_sha256": hash_file(out / "contention.json"),
                }
            )
        )


if __name__ == "__main__":
    main()
