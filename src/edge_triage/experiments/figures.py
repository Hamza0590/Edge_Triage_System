"""Nine scientific figures from verified, saved per-repetition metrics."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file


def scientific_plots(path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    value = json.loads(path.read_bytes())
    if value.get("schema_version") != 1 or value.get("purpose") not in {
        "unique_test",
        "unique_validation",
    }:
        raise ValueError("unsupported scientific summary schema")
    rows = [r for r in value["subruns"] if r["status"] == "complete" and r["metrics"]["science"]]
    if not rows:
        raise ValueError("no scientific results")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["case"], []).append(row["metrics"])
    directory = path.parent / f"plots_{hash_file(path)[:16]}"
    result: dict[str, Any] = {
        "schema_version": 1,
        "summary_sha256": hash_file(path),
        "plots": {},
        "scope": "descriptive repetitions; fixed-fit unique-row 95% bootstrap; no pooling",
    }

    def save(fig: Any, name: str) -> None:
        fig.supxlabel(
            f"n={rows[0]['metrics']['occurrences']} unique rows/run; "
            f"{min(len(r) for r in groups.values())}-{max(len(r) for r in groups.values())} "
            "timing repetitions/case; descriptive comparisons",
            fontsize=8,
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, metadata={"Software": "edge-triage"})
        plt.close(fig)
        destination = directory / f"{name}.png"
        write_immutable(destination, buffer.getvalue())
        result["plots"][destination.name] = hash_file(destination)

    for name, xlabel in (
        ("survival_compute", "Instrumented stream wall time (ms / candidate; all services)"),
        ("survival_bytes", "Logical output (MB; no codec or network)"),
    ):
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        for i, (case, runs) in enumerate(groups.items()):
            for j, run in enumerate(runs):
                science = run["science"]
                x = (
                    run["stream_wall_ms"] / run["occurrences"]
                    if name == "survival_compute"
                    else science["output_bytes"] / 1e6
                )
                lo, hi = science["bootstrap"]["intervals"]["survival_recall"]
                y = science["survival_recall"]
                ax.errorbar(
                    x,
                    y,
                    yerr=[[max(0, y - lo)], [max(0, hi - y)]],
                    fmt="o" if science["equal_recall_eligible"] else "x",
                    color=f"C{i % 10}",
                    capsize=2,
                    alpha=0.7,
                    label=f"{case} ({len(runs)} runs)" if j == 0 else None,
                )
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
        ax.set(
            xlabel=xlabel,
            ylabel="Real survival recall",
            title=f"{value['purpose']}: per-run fixed-fit 95% CI; x marks below target",
        )
        ax.legend(fontsize=7, ncols=2, loc="center left", bbox_to_anchor=(1, 0.5))
        save(fig, name)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for key in ("B1", "B2", "B3", "fixed_tiny", "fixed_medium", "fixed_large", "B5", "B6", "B7"):
        if key not in groups:
            continue
        for i, run in enumerate(groups[key]):
            capacity = run["executor"].get("capacity", 1)
            ax.scatter(capacity, run["throughput_candidates_per_s"], label=key if i == 0 else None)
    ax.set(
        xlabel="Executor capacity (jobs)",
        ylabel="Throughput (candidates / s)",
        title="Warmed run repetitions; capacity is not observed parallelism",
    )
    ax.legend(fontsize=8, ncols=3)
    save(fig, "throughput_concurrency")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for key in ("B5_arrival_20", "B5_arrival_100", "B5"):
        if key not in groups:
            continue
        for i, run in enumerate(groups[key]):
            rate = run["workload"]["realized_arrival_rate_per_s"]
            ax.scatter(
                rate, run["latency_ms"]["end_to_end_ms"]["p95"], label=key if i == 0 else None
            )
    ax.set(
        xlabel="Realized arrival rate (candidates / s; backpressure affected)",
        ylabel="P95 end-to-end (ms)",
        title="Same adaptive sequential system; matched validation fits per offered trace",
    )
    ax.legend()
    save(fig, "p95_arrival")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    for ax, keys in zip(axes, [("B5", "B6", "B7", "FULL"), ("B5_arrival_20", "B5_arrival_100")]):
        for key in keys:
            if key in groups:
                points = groups[key][0]["queue_series"]
                ax.plot(
                    [s["elapsed_s"] for s in points],
                    [s["length"] for s in points],
                    label=f"{key}, repetition 0",
                )
        ax.set(xlabel="Elapsed monitor time (s)", ylabel="Queue length (candidates)")
        ax.legend(fontsize=8)
    axes[0].set_title("Sampled queue trajectories; all repetitions retained in summary")
    save(fig, "queue_time")

    names = list(groups)
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    bottom = np.zeros(len(names))
    for tier in ("tiny", "medium", "large", "failed"):
        heights = np.array([groups[k][0]["tier_usage"].get(tier, 0) for k in names])
        ax.bar(names, heights, bottom=bottom, label=tier)
        bottom += heights
    ax.tick_params(axis="x", rotation=80, labelsize=8)
    ax.set(
        ylabel="Candidates (first repetition)",
        title="Final trusted tier use; complete per-run counts in summary",
    )
    ax.legend()
    save(fig, "tier_usage")

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for key in ("B4a", "B4b", "B4c", "B4d", "FULL"):
        if key not in groups:
            continue
        samples = groups[key][0]["uncertainty"]["samples"]
        for error, marker in ((0, "o"), (1, "x")):
            points = [s["std"] for s in samples if s["error"] == error]
            if points:
                ax.scatter(
                    points,
                    [key + (" error" if error else " correct")] * len(points),
                    marker=marker,
                    alpha=0.35,
                    s=18,
                    label=f"{key}: {len(points)} visits",
                )
    ax.set(
        xlabel="MC probability standard deviation",
        title="First repetition: sampled ambiguous tier visits; visits are not independent rows",
    )
    save(fig, "uncertainty_error")

    for name, selected in (
        ("threshold_comparison", ["B5_common", "B5"]),
        (
            "portfolio_comparison",
            [
                "portfolio_tiny",
                "portfolio_medium",
                "portfolio_large",
                "portfolio_tiny+medium",
                "portfolio_tiny+large",
                "portfolio_medium+large",
                "B5",
            ],
        ),
    ):
        selected = [k for k in selected if k in groups]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for i, key in enumerate(selected):
            for j, run in enumerate(groups[key]):
                science = run["science"]
                axes[0].scatter(i, science["survival_recall"], color="#247ba0")
                axes[1].scatter(i, science["output_bytes"] / 1e6, color="#be6a15")
                if j == 0:
                    lo, hi = science["bootstrap"]["intervals"]["survival_recall"]
                    axes[0].vlines(i, lo, hi, color="#247ba0")
        for ax in axes:
            ax.set_xticks(
                range(len(selected)),
                [f"{k}\n{len(groups[k])} runs" for k in selected],
                rotation=45,
                ha="right",
                fontsize=8,
            )
        axes[0].axhline(0.95, linestyle="--", color="black")
        axes[0].set_ylabel("Survival recall (first-run 95% CI)")
        axes[1].set_ylabel("Logical output (MB)")
        fig.suptitle("Frozen unique rows; below-target costs excluded from equal-recall claims")
        save(fig, name)
    component_rows = [
        r for r in value["subruns"] if r["status"] == "complete" and r["metrics"].get("component")
    ]
    if component_rows:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, tier in zip(axes, ("tiny", "medium", "large")):
            for backend, marker in (("microbatch", "o"), ("batch1_threads", "x")):
                selected_components = [
                    r
                    for r in component_rows
                    if r["case"].startswith(f"component_{tier}_{backend}_")
                ]
                ax.scatter(
                    [r["metrics"]["component"]["level"] for r in selected_components],
                    [r["metrics"]["throughput_candidates_per_s"] for r in selected_components],
                    marker=marker,
                    label=backend,
                )
            ax.set(title=tier, xlabel="Batch size or concurrent workers", ylabel="Candidates / s")
            ax.legend(fontsize=8)
        fig.suptitle(
            "Matched preprocessed components; individual repetitions; excludes pipeline services"
        )
        save(fig, "component_comparison")
    seed_cases = [k for k in groups if k.startswith("seed_")]
    if seed_cases:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for i, key in enumerate(seed_cases):
            science = groups[key][0]["science"]
            lo, hi = science["bootstrap"]["intervals"]["survival_recall"]
            axes[0].scatter(i, science["survival_recall"])
            axes[0].vlines(i, lo, hi)
            axes[1].scatter(i, science["output_bytes"] / 1e6)
        for ax in axes:
            ax.set_xticks(range(len(seed_cases)), seed_cases, rotation=65, ha="right", fontsize=8)
        axes[0].axhline(0.95, linestyle="--", color="black")
        axes[0].set_ylabel("Survival recall; first-run 95% fixed-fit CI")
        axes[1].set_ylabel("Logical output (MB)")
        fig.suptitle("All three frozen seeds; descriptive n=3; no best-seed selection")
        save(fig, "seed_comparison")
    destination = directory / "index.json"
    write_immutable(destination, canonical(result))
    return destination
