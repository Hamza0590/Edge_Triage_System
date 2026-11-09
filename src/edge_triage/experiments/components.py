"""Matched component-only microbatch and concurrent batch-1 experiment backend."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from edge_triage.config import AppConfig
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.data.loaders import MeerlichtDataset
from edge_triage.experiments.promotion import verify_promotion
from edge_triage.experiments.reporting import distribution
from edge_triage.experiments.workloads import load_trace
from edge_triage.health import hash_file
from edge_triage.models.inference_state import INFERENCE_STATE
from edge_triage.models.registry import read_index
from edge_triage.runs import RunManager
from edge_triage.runtime.factory import assemble
from edge_triage.runtime.microbatch import replay_batches

COMPONENT_FILES = (
    "manifest.json",
    "config.resolved.yaml",
    "events.jsonl",
    "metrics/component.json",
)


def run_component(
    config: AppConfig, backend: str, level: int, warmup: int
) -> tuple[Path, dict[str, Any]]:
    if config.pipeline.partition == "test":
        verify_promotion(config)
    if (
        len(config.models.tier_portfolio) != 1
        or backend not in {"microbatch", "batch1_threads"}
        or config.training.device != "cpu"
        or level not in (1, 2, 4, 8)
    ):
        raise ValueError("component requires singleton and supported backend")
    assert config.workload.trace and config.split.manifest and config.models.checkpoint_index
    trace = load_trace(
        config.workload.trace, hash_file(config.split.manifest), partition=config.pipeline.partition
    )
    if any(r.relative_arrival_seconds != 0 for r in trace):
        raise ValueError("matched component comparison currently requires backlog")
    torch.set_num_threads(config.training.cpu_threads)
    with RunManager(config) as run:
        parts = assemble(config)
        tier = config.models.tier_portfolio[0]
        entry = read_index(config.models.checkpoint_index)["checkpoints"][tier.value]
        run.register_checkpoint(
            Path(entry["directory"]) / "checkpoint_manifest.json", entry["checkpoint_id"]
        )
        adapter: Any = parts.models[tier]
        model = getattr(adapter, "base", adapter).model
        dataset = MeerlichtDataset(config, config.pipeline.partition, run_id=run.manifest.run_id)
        positions = {row["candidate_id"]: i for i, row in enumerate(dataset.rows)}
        started = time.perf_counter_ns()
        images = np.stack([dataset[positions[r.source_candidate_id]][1].numpy() for r in trace])
        preprocessing_ms = (time.perf_counter_ns() - started) / 1e6
        with INFERENCE_STATE.read(), torch.inference_mode():
            for _ in range(warmup):
                model(torch.from_numpy(images[:level].copy()))

        def infer(index: int, origin: int) -> dict[str, Any]:
            start = time.perf_counter_ns()
            with INFERENCE_STATE.read(), torch.inference_mode():
                logit = float(model(torch.from_numpy(images[index : index + 1].copy())).item())
            end = time.perf_counter_ns()
            return {
                "position": index,
                "logit": logit,
                "wait_ms": (start - origin) / 1e6,
                "inference_ms": (end - start) / 1e6,
                "end_to_end_ms": (end - origin) / 1e6,
            }

        if backend == "microbatch":
            result = replay_batches(model, images, [0.0] * len(images), level, 0.002)
        else:
            with ThreadPoolExecutor(level) as pool:
                list(pool.map(lambda i: infer(i, time.perf_counter_ns()), range(level)))
                origin = time.perf_counter_ns()
                futures = [pool.submit(infer, i, origin) for i in range(len(images))]
                rows = [f.result() for f in futures]
                elapsed = (time.perf_counter_ns() - origin) / 1e9
                result = {
                    "rows": rows,
                    "elapsed_s": elapsed,
                    "throughput_per_s": len(rows) / elapsed,
                }
        if [r["position"] for r in result["rows"]] != list(range(len(images))) or not np.isfinite(
            [r["logit"] for r in result["rows"]]
        ).all():
            raise ValueError("component lost identity or finite output")
        for row, source in zip(result["rows"], trace, strict=True):
            row["candidate_id"] = source.source_candidate_id
        result.update(
            backend=backend,
            level=level,
            preprocessing_ms=preprocessing_ms,
            warmup_forwards=warmup,
            worker_warmup_forwards=level if backend == "batch1_threads" else 0,
            trace_sha256=hash_file(config.workload.trace),
            scope="preprocessed component only; no monitor/scheduler/fate/logging in timed region",
        )
        write_immutable(run.directory / "metrics/component.json", canonical(result))
    return run.directory, inspect_component(config, run.directory, backend, level, warmup)


def inspect_component(
    config: AppConfig, run: Path, backend: str, level: int, warmup: int
) -> dict[str, Any]:
    """Regenerate component metrics from sealed outputs, without new inference."""
    from edge_triage.config import load_config

    manifest = json.loads((run / "manifest.json").read_bytes())
    if (
        manifest["status"] != "completed"
        or manifest["config_hash"] != config.config_hash
        or load_config(run / "config.resolved.yaml").config_hash != config.config_hash
    ):
        raise ValueError("component runtime/configuration mismatch")
    assert config.workload.trace and config.split.manifest
    trace = load_trace(
        config.workload.trace, hash_file(config.split.manifest), partition=config.pipeline.partition
    )
    result = json.loads((run / "metrics/component.json").read_bytes())
    rows = result["rows"]
    if (
        result["backend"] != backend
        or result["level"] != level
        or result["warmup_forwards"] != warmup
        or result["trace_sha256"] != hash_file(config.workload.trace)
        or [r["position"] for r in rows] != list(range(len(trace)))
        or [r["candidate_id"] for r in rows] != [r.source_candidate_id for r in trace]
        or not np.isfinite([r["logit"] for r in rows]).all()
        or result["elapsed_s"] <= 0
        or not np.isclose(result["throughput_per_s"], len(rows) / result["elapsed_s"])
    ):
        raise ValueError("component output identity/protocol mismatch")
    return {
        "schema_version": 1,
        "occurrences": len(rows),
        "unique_sources": len({r.source_candidate_id for r in trace}),
        "science": None,
        "science_scope": "component_only_no_equal_recall_cost_claim",
        "failures": 0,
        "throughput_candidates_per_s": result["throughput_per_s"],
        "stream_wall_ms": result["elapsed_s"] * 1000,
        "latency_ms": {
            "end_to_end_ms": distribution([r["end_to_end_ms"] for r in result["rows"]]),
            "queue_ms": distribution([r["wait_ms"] for r in result["rows"]]),
        },
        "component": {"backend": backend, "level": level, "scope": result["scope"]},
        "test_evaluated": config.pipeline.partition == "test",
    }
