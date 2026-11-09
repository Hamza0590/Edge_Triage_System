"""Saved-record metrics with explicit sample scope and overlapping cost boundaries."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np

from edge_triage.calibration.metrics import bootstrap, metrics
from edge_triage.runtime.records import CandidateRecord


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("finite nonnegative timing/resource values required")
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": max(values),
    }


def latency_partition(row: CandidateRecord) -> dict[str, float]:
    """Disjoint coarse candidate clock regions; fine costs overlap these regions."""
    boundaries = [
        row.arrival_ns,
        row.queue_ns,
        row.final_ns if row.start_ns is None else row.start_ns,
        row.final_ns if row.end_ns is None else row.end_ns,
        row.final_ns,
    ]
    if boundaries != sorted(boundaries):
        raise ValueError("invalid candidate clock partition")
    names = (
        "arrival_to_queue_ms",
        "queue_to_first_start_ms",
        "first_start_to_last_end_ms",
        "last_end_to_final_ms",
    )
    result = {name: (b - a) / 1e6 for name, a, b in zip(names, boundaries, boundaries[1:])}
    if not math.isclose(sum(result.values()), row.end_to_end_ms, abs_tol=1e-6):
        raise ValueError("candidate latency component sum mismatch")
    return result


def science_metrics(rows: list[CandidateRecord], target: float) -> dict[str, Any]:
    real = [r for r in rows if r.label == 1]
    discarded = sum(r.fate == "discard" for r in real)
    survival = (len(real) - discarded) / len(real) if real else None
    valid = [r for r in rows if r.final_trusted_p_real is not None and r.status == "completed"]
    # Use final trusted probabilities, preserving raw/calibrated mode per configuration.
    probabilities = np.clip([r.final_trusted_p_real for r in valid], 1e-15, 1 - 1e-15)
    scores = np.log(probabilities) - np.log1p(-probabilities)
    classifier = metrics([r.label for r in valid], scores) if valid else None
    input_bytes, output_bytes = sum(r.input_bytes for r in rows), sum(r.output_bytes for r in rows)
    return {
        "n": len(rows),
        "real": len(real),
        "survival_recall": survival,
        "false_real_discards": discarded,
        "real_transmit_recall": sum(r.fate == "transmit" for r in real) / len(real)
        if real
        else None,
        "real_compress_recall": sum(r.fate == "compress" for r in real) / len(real)
        if real
        else None,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "byte_reduction": 1 - output_bytes / input_bytes if input_bytes else None,
        "classifier": classifier,
        "classification_missing": len(rows) - len(valid),
        "target": target,
        "equal_recall_eligible": survival is not None and survival >= target,
        "target_status": "undefined"
        if survival is None
        else "meets_empirical_target"
        if survival >= target
        else "below_target_no_equal_recall_claim",
        "threshold_refitted": False,
    }


def summarize_records(
    rows: list[CandidateRecord],
    sources: dict[str, str],
    runtime: dict[str, Any],
    resources: list[dict[str, Any]],
    *,
    unique_validation: bool,
    target: float,
    bootstrap_repeats: int,
    seed: int,
    partition: str = "validation",
) -> dict[str, Any]:
    if not rows or len({r.candidate_id for r in rows}) != len(rows):
        raise ValueError("empty results or duplicate occurrence IDs")
    if set(sources) != {r.candidate_id for r in rows}:
        raise ValueError("result/trace identity mismatch")
    if any(r.fate not in {"discard", "compress", "transmit"} for r in rows):
        raise ValueError("missing or unrecognized fate in experiment output")
    unique_count = len(set(sources.values()))
    if unique_validation and unique_count != len(rows):
        raise ValueError("replayed duplicates cannot become independent classification samples")
    partitions = [latency_partition(r) for r in rows]
    fine_names = (
        "inference_ms",
        "uq_ms",
        "admission_ms",
        "executor_ms",
        "escalation_decision_ms",
        "fate_ms",
        "compression_cpu_ms",
    )
    fine = {name: sum(getattr(r, name) for r in rows) for name in fine_names}
    forward_by_tier: Counter[str] = Counter()
    initial_ms = escalation_ms = 0.0
    for row in rows:
        chain = json.loads(row.execution_chain_json)
        if chain:
            initial_ms += chain[0]["inference_ms"]
            escalation_ms += sum(job["inference_ms"] for job in chain[1:])
        for job in chain:
            # Count attempted deterministic forwards, including failures after launch.
            forward_by_tier[job["tier"]] += int(job["started_ns"] is not None)
            forward_by_tier[job["tier"]] += job["additional_forward_passes"]
    scientific: dict[str, Any] | None = None
    if unique_validation:
        scientific = science_metrics(rows, target)
        valid = [r for r in rows if r.final_trusted_p_real is not None and r.status == "completed"]
        if valid:
            probabilities = np.clip([r.final_trusted_p_real for r in valid], 1e-15, 1 - 1e-15)
            scientific["classification_bootstrap"] = bootstrap(
                [r.label for r in valid],
                np.log(probabilities) - np.log1p(-probabilities),
                temperature=1,
                threshold=0,
                bins=15,
                repeats=bootstrap_repeats,
                seed=seed,
            )
            scientific["classification_bootstrap"]["limitation"] = (
                f"unique {partition} rows; fixed fit; no group/selection/seed uncertainty"
            )
        rng = np.random.default_rng(seed)
        intervals: dict[str, list[float]] = {k: [] for k in ("survival_recall", "byte_reduction")}
        for _ in range(bootstrap_repeats):
            sampled = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
            real = [r for r in sampled if r.label]
            if real:
                intervals["survival_recall"].append(
                    sum(r.fate != "discard" for r in real) / len(real)
                )
            original = sum(r.input_bytes for r in sampled)
            if original:
                intervals["byte_reduction"].append(
                    1 - sum(r.output_bytes for r in sampled) / original
                )
        scientific["bootstrap"] = {
            "repeats": bootstrap_repeats,
            "seed": seed,
            "scope": f"unique-row fixed-fit {partition}; no independence guarantee",
            "intervals": {
                k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else None
                for k, v in intervals.items()
            },
        }
    stream_ms = runtime["stream_wall_ms"]
    return {
        "schema_version": 1,
        "occurrences": len(rows),
        "unique_sources": unique_count,
        "repeated_occurrences": len(rows) - unique_count,
        "science": scientific,
        "science_scope": f"unique_{partition}" if unique_validation else "systems_smoke_only",
        "failures": sum(r.status == "failed" for r in rows),
        "oom_failures": sum("outofmemory" in (r.error_code or "").lower() for r in rows),
        "stream_wall_ms": stream_ms,
        "throughput_candidates_per_s": len(rows) * 1000 / stream_ms if stream_ms else None,
        "latency_ms": {
            key: distribution([getattr(r, key) for r in rows])
            for key in ("queue_ms", "inference_ms", "end_to_end_ms")
        },
        "costs": {
            "candidate_clock_partition_ms": {
                k: sum(p[k] for p in partitions) for k in partitions[0]
            },
            "candidate_end_to_end_sum_ms": sum(r.end_to_end_ms for r in rows),
            "fine_components_ms": {
                **fine,
                "initial_inference_ms": initial_ms,
                "escalation_inference_ms": escalation_ms,
                "preprocessing_ms": runtime["workload"]["preprocessing_ms"],
                "monitor_ms": runtime["monitor"]["sampling_seconds"] * 1000,
                "selection_ms": runtime["scheduler"]["selection_total_ms"],
            },
            "fine_components_additive": False,
            "scope": (
                "fine costs overlap clock regions and concurrent work; never sum into wall time"
            ),
            "transmission_ms": None,
            "transmission_scope": "logical bytes only; no network executes",
        },
        "forward_passes_by_tier": dict(forward_by_tier),
        "total_forward_passes": sum(forward_by_tier.values()),
        "average_forward_passes": sum(forward_by_tier.values()) / len(rows),
        "tier_usage": dict(Counter(r.final_trusted_tier or "failed" for r in rows)),
        "uncertainty": uncertainty_metrics(rows),
        "queue_series": [
            {
                "elapsed_s": (s["monotonic_ns"] - resources[0]["monotonic_ns"]) / 1e9,
                "length": s["queue_length"],
            }
            for s in resources
        ],
        "queue_growth_per_s": (
            (resources[-1]["queue_length"] - resources[0]["queue_length"])
            * 1e9
            / (resources[-1]["monotonic_ns"] - resources[0]["monotonic_ns"])
            if len(resources) > 1 and resources[-1]["monotonic_ns"] > resources[0]["monotonic_ns"]
            else None
        ),
        "workload": runtime["workload"],
        "resources": {
            key: distribution([float(s[key]) for s in resources if s.get(key) is not None])
            for key in (
                "cpu_percent",
                "process_rss_bytes",
                "gpu_utilization_percent",
                "gpu_memory_used_bytes",
                "queue_length",
                "active_jobs_total",
            )
        },
        "queue": runtime["queue"],
        "executor": runtime["executor"],
        "warmup": runtime["warmup"],
        "test_evaluated": partition == "test",
    }


def uncertainty_metrics(rows: list[CandidateRecord]) -> dict[str, Any]:
    samples = []
    escalated = corrections = regressions = 0
    for row in rows:
        predictions = json.loads(row.prediction_chain_json)
        uq = json.loads(row.uncertainty_chain_json)
        if len(predictions) > 1:
            escalated += 1
            first = predictions[0]
            score = first.get("calibrated_p_real")
            score = first["p_real"] if score is None else score
            initial_correct = (score >= 0.5) == bool(row.label)
            final_correct = row.final_trusted_p_real is not None and (
                row.final_trusted_p_real >= 0.5
            ) == bool(row.label)
            corrections += int(not initial_correct and final_correct)
            regressions += int(initial_correct and not final_correct)
        for prediction, uncertainty in zip(predictions, uq):
            if uncertainty.get("additional_forward_passes", 0):
                score = prediction.get("calibrated_p_real")
                score = prediction["p_real"] if score is None else score
                samples.append(
                    {
                        "candidate_id": row.candidate_id,
                        "std": uncertainty["standard_deviation"],
                        "error": int((score >= 0.5) != bool(row.label)),
                    }
                )
    error_auc = None
    if samples and len({s["error"] for s in samples}) == 2:
        error_auc = metrics([s["error"] for s in samples], [s["std"] for s in samples])["roc_auc"]
    return {
        "escalated": escalated,
        "corrections": corrections,
        "regressions": regressions,
        "escalation_rate": escalated / len(rows),
        "correction_rate": corrections / len(rows),
        "regression_rate": regressions / len(rows),
        "error_auroc": error_auc,
        "scope": "sampled ambiguous tier visits only; descriptive; no independent-visit CI",
        "samples": samples,
    }
