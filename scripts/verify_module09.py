"""Verify Module09 evidence, replay measured dynamic caps, and plot saved curves."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import threading
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from benchmark_module09 import inspect_run, quantiles
from preflight_module09 import verify

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import hash_file
from edge_triage.runtime.streaming import TraceSource, run_concurrent
from edge_triage.scheduling.contention import load_contention_cap

ROOT = Path(__file__).resolve().parents[1]


def fixed(out):
    suite = json.loads((ROOT / "artifacts/traces/suite_a515d472c9d5ef05.json").read_bytes())
    trace_path = Path(
        next(
            r["path"]
            for r in suite["traces"]
            if r["profile"] == "backlog" and r["process"] == "periodic"
        )
    )
    config = load_config(ROOT / "configs/experiments/module09_v1.yaml")
    trace = load_trace(trace_path, hash_file(config.split.manifest))
    count = rounds = 0
    paths = sorted(out.glob("pipeline_*.json"))
    assert len(paths) == 28
    for path in paths:
        setting = json.loads(path.read_bytes())
        if setting["skipped"]:
            continue
        assert len(setting["rounds"]) == 3
        for saved in setting["rounds"]:
            directory = Path(saved["run"])
            manifest = json.loads((directory / "manifest.json").read_bytes())
            assert manifest["status"] == "completed"
            assert manifest["arrival_trace_hash"]["digest"] == hash_file(trace_path)
            current, _ = inspect_run(directory, trace)
            assert {k: v for k, v in saved.items() if k != "round"} == current
            assert current["failed"] == 0
            count += current["count"]
            rounds += 1
    write_immutable(
        out / "fixed_verification.json",
        canonical(
            {
                "rounds": rounds,
                "completed": count,
                "failed": 0,
                "test_evaluated": False,
                "settings": {p.name: hash_file(p) for p in paths},
            }
        ),
    )
    print(f"Verified {rounds} fixed rounds / {count} candidate records", flush=True)


def components(out):
    files = sorted(p for p in out.glob("component_*.json") if not p.name.endswith("_failure.json"))
    assert len(files) == 48, len(files)
    config = load_config(ROOT / "configs/experiments/module09_v1.yaml")
    suite = json.loads((ROOT / "artifacts/traces/suite_a515d472c9d5ef05.json").read_bytes())
    traces = {
        entry["profile"]: load_trace(Path(entry["path"]), hash_file(config.split.manifest))
        for entry in suite["traces"]
        if entry["process"] == "periodic" and entry["profile"] in ("backlog", "moderate")
    }
    identities = [(r.candidate_id, r.dataset_index) for r in traces["backlog"]]
    assert identities == [(r.candidate_id, r.dataset_index) for r in traces["moderate"]]
    source = TraceSource(config, "component-verification", traces["backlog"], threading.Event())
    images = np.load(out / "component_images.npy", mmap_mode="r")
    for image, record in zip(images, traces["backlog"], strict=True):
        np.testing.assert_array_equal(
            image, source.dataset[source.positions[record.dataset_index]][1].numpy()
        )
    for workload, records in traces.items():
        assert json.loads((out / f"arrivals_{workload}.json").read_bytes()) == [
            r.relative_arrival_seconds for r in records
        ]
    summaries = []
    count = 0
    for tier in ModelTier:
        reference = json.loads(
            (out / f"component_{tier.value}_threads_1_backlog.json").read_bytes()
        )
        expected = {r["position"]: r["logit"] for r in reference["rounds"][0]["rows"]}
        for path in (p for p in files if p.name.startswith(f"component_{tier.value}_")):
            data = json.loads(path.read_bytes())
            if data.get("skipped"):
                summaries.append({"file": path.name, "skipped": True, "reason": data["reason"]})
                continue
            workload = "moderate" if path.name.endswith("moderate.json") else "backlog"
            assert data["inputs"] == {
                "config_sha256": hash_file(ROOT / "configs/experiments/module09_v1.yaml"),
                "images_sha256": hash_file(out / "component_images.npy"),
                "arrivals_sha256": hash_file(out / f"arrivals_{workload}.json"),
                "script_sha256": hash_file(ROOT / "scripts/module09_components.py"),
            }
            assert len(data["rounds"]) == 3
            rows_summary = []
            for row in data["rounds"]:
                actual = {r["position"]: r["logit"] for r in row["rows"]}
                assert len(row["rows"]) == len(expected) and set(actual) == set(expected)
                np.testing.assert_allclose(
                    [actual[k] for k in sorted(expected)],
                    [expected[k] for k in sorted(expected)],
                    rtol=1e-5,
                    atol=1e-5,
                )
                assert all((actual[k] >= 0) == (expected[k] >= 0) for k in expected)
                assert all(r["end_to_end_ms"] >= r["wait_ms"] >= 0 for r in row["rows"])
                stats = {
                    "throughput_per_s": row["throughput_per_s"],
                    "wait_ms": quantiles([r["wait_ms"] for r in row["rows"]]),
                    "end_to_end_ms": quantiles([r["end_to_end_ms"] for r in row["rows"]]),
                    "aggregate_worker_rss": row.get("aggregate_worker_rss"),
                }
                if "batches" in row:
                    assert sum(b["size"] for b in row["batches"]) == len(expected)
                    assert all(1 <= b["size"] <= data["level"] for b in row["batches"])
                    stats["forward_ms_per_candidate"] = sum(
                        b["forward_ms"] for b in row["batches"]
                    ) / len(expected)
                    stats["mean_batch_size"] = len(expected) / len(row["batches"])
                else:
                    stats["inference_ms"] = quantiles([r["inference_ms"] for r in row["rows"]])
                rows_summary.append(stats)
                count += len(actual)
            summaries.append(
                {
                    "file": path.name,
                    "tier": tier.value,
                    "backend": data["backend"],
                    "level": data["level"],
                    "startup_s": data["startup_s"],
                    "rounds": rows_summary,
                    "sha256": hash_file(path),
                    "skipped": False,
                }
            )
    report = {
        "verified_candidates": count,
        "settings": summaries,
        "logit_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "same_zero_logit_classifications": True,
        "test_evaluated": False,
        "images_sha256": hash_file(out / "component_images.npy"),
        "validation_identity_order": identities,
        "scope": (
            "preprocessed component timing; includes task scheduling/IPC and RSS sampling; "
            "excludes durable pipeline events and preprocessing"
        ),
    }
    write_immutable(out / "components_verification.json", canonical(report))
    print(f"Verified {len(summaries)} component settings / {count} candidate outputs", flush=True)


def replay(out):
    base = load_config(ROOT / "configs/experiments/module09_v1.yaml")
    raw = base.model_dump(mode="json")
    raw["admission"].update(policy="memory_contention_aware", execution_mode="dynamic_concurrency")
    raw["scheduler"].update(
        contention_profile=str(out / "contention.json"),
        contention_sha256=hash_file(out / "contention.json"),
    )
    raw["experiment"]["name"] = "module09_proposed_v2"
    proposed = ROOT / "configs/experiments/module09_proposed_v2.yaml"
    write_immutable(proposed, yaml.safe_dump(raw, sort_keys=False).encode())
    config = load_config(proposed)
    benchmark = json.loads((out / "benchmark.json").read_bytes())
    assert (
        hash_file(out / "benchmark.json")
        == json.loads((out / "contention.json").read_bytes())["benchmark_sha256"]
    )
    for name, digest in benchmark["files"].items():
        assert hash_file(out / name) == digest, name
    for name, digest in json.loads((out / "measurement_source.json").read_bytes()).items():
        assert hash_file(ROOT / name) == digest, name
    suite = json.loads((ROOT / "artifacts/traces/suite_a515d472c9d5ef05.json").read_bytes())
    traces = {r["profile"]: Path(r["path"]) for r in suite["traces"] if r["process"] == "periodic"}
    conditions = [
        (p, "backlog", "round_robin")
        for n in (1, 2, 3)
        for p in itertools.combinations(ModelTier, n)
    ] + [
        (tuple(ModelTier), name, "resource_queue_aware") for name in ("low", "burst", "alternating")
    ]
    results = []
    for portfolio, workload, policy in conditions:
        key = "+".join(t.value for t in portfolio)
        path = out / f"dynamic_{key}_{workload}.json"
        trace = load_trace(traces[workload], hash_file(config.split.manifest))
        update = config.model_dump()
        update["models"]["tier_portfolio"] = portfolio
        update["tier_selection"].update(policy=policy, fixed_tier=portfolio[0])
        update["workload"]["trace"] = traces[workload]
        selected = AppConfig.model_validate(update)
        if path.exists():
            saved = json.loads(path.read_bytes())
            directory = Path(saved["run"])
        else:
            with open(os.devnull, "w") as console, contextlib.redirect_stderr(console):
                directory = run_concurrent(selected)
        result, actual = inspect_run(directory, trace)
        assert result["failed"] == 0
        assert result["executor"]["capacity"] == load_contention_cap(selected)
        assert all(r["additional_forward_passes"] == 0 for r in actual.values())
        if workload == "backlog":
            baseline = json.loads((out / f"pipeline_{key}_1.json").read_bytes())
            expected = {
                r["candidate_id"]: r
                for r in pq.read_table(
                    Path(baseline["rounds"][0]["run"]) / "candidates.parquet"
                ).to_pylist()
            }
            for identity, row in actual.items():
                reference = expected[identity]
                assert row["selected_tiers"] == reference["selected_tiers"]
                assert abs(row["final_logit"] - reference["final_logit"]) < 1e-5
                assert abs(row["final_p_real"] - reference["final_p_real"]) < 1e-6
                assert (row["final_p_real"] >= 0.5) == (reference["final_p_real"] >= 0.5)
        result.update(portfolio=key, workload_profile=workload, tier_policy=policy)
        write_immutable(path, canonical(result))
        results.append(result)
        print(
            f"dynamic {key} {workload}: {result['count']} completed, "
            f"cap {result['executor']['capacity']}",
            flush=True,
        )
    write_immutable(
        out / "verification.json",
        canonical(
            {
                "runs": results,
                "completed": sum(r["count"] for r in results),
                "failed": 0,
                "gate_d_pass": benchmark["gate_d_pass"],
                "caps": benchmark["caps"],
                "preflight": verify(),
                "test_evaluated": False,
                "contention_sha256": hash_file(out / "contention.json"),
                "config_sha256": hash_file(proposed),
            }
        ),
    )


def plots(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [json.loads(p.read_bytes()) for p in sorted(out.glob("pipeline_*.json"))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key in sorted({r["portfolio"] for r in results}):
        group = sorted(
            [r for r in results if r["portfolio"] == key and not r["skipped"]],
            key=lambda r: r["level"],
        )
        levels = [r["level"] for r in group]
        for ax, name, field in zip(
            axes.flat,
            ("Throughput (candidates/s)", "E2E P95 (ms)", "Inference P95 (ms)", "Queue P95 (ms)"),
            ("throughput_per_s", "end_to_end_ms", "inference_ms", "queue_ms"),
            strict=True,
        ):
            values = [
                [r[field] if field == "throughput_per_s" else r[field]["p95"] for r in g["rounds"]]
                for g in group
            ]
            ax.plot(levels, [np.median(v) for v in values], marker="o", label=key)
            ax.fill_between(levels, [min(v) for v in values], [max(v) for v in values], alpha=0.08)
            ax.set(xlabel="Configured concurrency", ylabel=name, xticks=[1, 2, 4, 8])
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Module09 CPU pipeline: median and range of three rounds\n"
        "120 validation occurrences; full durable instrumentation; no UQ"
    )
    directory = out / "plots"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "fixed_curves.png", dpi=160)
    plt.close(fig)
    data = json.loads((out / "components_verification.json").read_bytes())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, tier in zip(axes, ModelTier, strict=True):
        for backend in ("threads", "processes", "microbatch"):
            group = sorted(
                [
                    r
                    for r in data["settings"]
                    if not r["skipped"]
                    and r["tier"] == tier.value
                    and r["backend"] == backend
                    and r["file"].endswith("backlog.json")
                ],
                key=lambda r: r["level"],
            )
            ax.plot(
                [r["level"] for r in group],
                [np.median([v["throughput_per_s"] for v in r["rounds"]]) for r in group],
                marker="o",
                label=backend,
            )
        ax.set(
            title=tier.value.capitalize(),
            xlabel="Workers / batch size",
            ylabel="Candidates/s",
            xticks=[1, 2, 4, 8],
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        "Preprocessed backlog components: three-round median\n"
        "Component scope excludes durable pipeline instrumentation"
    )
    fig.savefig(directory / "components.png", dpi=160)
    plt.close(fig)
    write_immutable(
        directory / "index.json",
        canonical(
            {
                "benchmark_sha256": hash_file(out / "benchmark.json"),
                "components_sha256": hash_file(out / "components_verification.json"),
                "plots": {p.name: hash_file(p) for p in directory.glob("*.png")},
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage", choices=["fixed", "components", "replay", "plots"], required=True
    )
    args = parser.parse_args()
    if args.stage == "replay":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    globals()[args.stage](Path(args.output).resolve())
