"""Fit and benchmark Module08 on frozen validation rows only; unique immutable outputs."""

import argparse
import itertools
import json
import time
import uuid
from pathlib import Path

import matplotlib
import numpy as np
import torch
from preflight_module07 import verify as preflight

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier, utc_now
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.data.loaders import MeerlichtDataset
from edge_triage.health import dataset_hashes, environment_info, hash_file
from edge_triage.runtime.factory import assemble
from edge_triage.runtime.interfaces import CandidateInput
from edge_triage.uncertainty.analysis import (
    cascade_analysis,
    fit_interval,
    fit_std,
    probability_metrics,
    separation,
)
from edge_triage.uncertainty.artifacts import TierUncertainty, UncertaintyArtifact
from edge_triage.uncertainty.policies import diagnostic_seed
from edge_triage.uncertainty.sampling import sample_probabilities, summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    write_immutable(path, canonical(value))


def benchmark(output):
    evidence = preflight()
    objective_path = ROOT / "configs/experiments/module08_objective_v1.json"
    protocol_path = ROOT / "configs/experiments/module08_analysis_protocol_v1.json"
    objective = json.loads(objective_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    config = load_config(ROOT / "configs/experiments/module07_v1.yaml")
    raw = config.model_dump()
    raw["calibration"]["probability_mode"] = objective["probability_mode"]
    config = AppConfig.model_validate(raw)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    identifier = utc_now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
    directory = output / identifier
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "objective.json", objective)
    save(directory / "protocol.json", protocol)
    save(directory / "preflight.json", evidence)
    write_immutable(directory / "config.resolved.yaml", config.resolved_yaml().encode())
    save(directory / "environment.json", environment_info())
    parts = assemble(config)
    dataset = MeerlichtDataset(config, "validation", run_id=identifier)
    samples = [dataset[i] for i in range(len(dataset))]
    assert len(samples) == 643 and all(s[0].label is None for s in samples)
    calibration = json.loads(config.calibration.artifact_index.read_text())
    metrics_path = config.calibration.artifact_index.parent / "metrics.json"
    saved_metrics = json.loads(metrics_path.read_text())
    assert hash_file(metrics_path) == (
        "c76cae0ab06790d523045275246f4ed57341647c2b23b1a8ed34a0518db8f30f"
    )
    all_rows = {n: {} for n in objective["passes"]}
    specs = {n: [] for n in objective["passes"]}
    tier_reports = {n: {} for n in objective["passes"]}
    calibration_baselines = {}
    for tier in ModelTier:
        adapter = parts.models[tier]
        entry = calibration["calibrations"][tier.value]
        assert hash_file(Path(entry["predictions"])) == entry["predictions_sha256"]
        frozen = json.loads(Path(entry["predictions"]).read_text())
        frozen = {r["candidate_id"]: r for r in frozen}
        assert set(frozen) == {s[0].candidate_id for s in samples}
        saved = [frozen[s[0].candidate_id] for s in samples]
        p = np.array([r["p_real_calibrated"] for r in saved])
        y = np.array([s[2] for s in samples])
        assert np.array_equal(y, [r["label"] for r in saved])
        lower, upper = fit_interval(p, objective["ambiguous_fraction"], partition="validation")
        for _ in range(20):
            adapter.predict(CandidateInput(samples[0][0], samples[0][1], 0, 120000), tier)
        for n in objective["passes"]:
            sample_probabilities(
                adapter.base.model,
                samples[0][1],
                n,
                temperature=adapter.artifact.temperature,
                seed=1,
            )
            all_rows[n][tier.value] = []
        for i, sample in enumerate(samples):
            candidate, image, label = sample[:3]
            item = CandidateInput(candidate, image, label, 120000)
            start = time.perf_counter_ns()
            pred = adapter.predict(item, tier)
            deterministic_ms = (time.perf_counter_ns() - start) / 1e6
            assert abs(pred.logit - saved[i]["logit"]) < 2e-5
            budgets = objective["passes"]
            budgets = budgets[i % 3 :] + budgets[: i % 3]
            for n in budgets:
                seed = diagnostic_seed(objective["seed"], candidate.candidate_id, tier)
                values, elapsed = sample_probabilities(
                    adapter.base.model,
                    image,
                    n,
                    temperature=adapter.artifact.temperature,
                    seed=seed,
                )
                uq = summarize(pred, values, elapsed)
                all_rows[n][tier.value].append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "label": label,
                        "tier": tier.value,
                        "checkpoint_id": pred.checkpoint_id,
                        "logit": saved[i]["logit"],
                        "observed_batch1_logit": pred.logit,
                        "p": saved[i]["p_real_calibrated"],
                        "raw_p": saved[i]["p_real_uncalibrated"],
                        "passes": n,
                        "samples": values,
                        "mean": uq.mean_p_real,
                        "std": uq.standard_deviation,
                        "entropy": uq.predictive_entropy,
                        "variation_ratio": uq.variation_ratio,
                        "seed": seed,
                        "deterministic_ms": deterministic_ms,
                        "mc_ms": elapsed,
                        "ambiguous": bool(lower <= p[i] <= upper),
                        "frozen_threshold_logit": saved_metrics[tier.value][
                            "equal_recall_raw_logit_threshold"
                        ],
                    }
                )
        calibration_baselines[tier.value] = {
            "raw": probability_metrics(y, [r["p_real_uncalibrated"] for r in saved]),
            "calibrated": probability_metrics(y, p),
            "module05_recommended_mode": adapter.artifact.recommended_mode,
        }
        for n in objective["passes"]:
            rows = all_rows[n][tier.value]
            threshold = fit_std(
                [r["std"] for r in rows if r["ambiguous"]],
                objective["std_quantile"],
                partition="validation",
            )
            for r in rows:
                r["unresolved"] = r["std"] >= threshold or (r["mean"] >= 0.5) != (r["p"] >= 0.5)
            specs[n].append(
                TierUncertainty(
                    tier=tier,
                    checkpoint_id=adapter.base.checkpoint_id,
                    calibration_artifact_id=adapter.artifact.artifact_id,
                    calibration_sha256=entry["sha256"],
                    probability_mode="calibrated",
                    temperature=adapter.artifact.temperature,
                    lower=lower,
                    upper=upper,
                    std_threshold=threshold,
                )
            )
            selective = [r["std"] if r["ambiguous"] else 0 for r in rows]
            tier_reports[n][tier.value] = {
                "lower": lower,
                "upper": upper,
                "std_threshold": threshold,
                "ambiguous_count": sum(r["ambiguous"] for r in rows),
                "selective": separation(y, p, selective),
                "always": separation(y, p, [r["std"] for r in rows]),
                "mc_mean_calibration": probability_metrics(y, [r["mean"] for r in rows]),
                "selective_mean_calibration": probability_metrics(
                    y, [r["mean"] if r["ambiguous"] else r["p"] for r in rows]
                ),
            }
        print(f"{tier.value}: {len(samples)} validation candidates x5/10/20 complete", flush=True)
    artifacts, analyses, gates = {}, {}, {}
    portfolios = [
        p for n in (1, 2, 3) for p in itertools.combinations([t.value for t in ModelTier], n)
    ]
    for n in objective["passes"]:
        artifact = UncertaintyArtifact(
            artifact_id=f"selective_mc_v1:{identifier}:{n}",
            passes=n,
            tiers=tuple(specs[n]),
            split_sha256=hash_file(config.split.manifest),
            preprocessing_sha256=hash_file(config.preprocessing.artifact),
            family_sha256=hash_file(config.models.checkpoint_index),
            calibration_index_sha256=hash_file(config.calibration.artifact_index),
            objective=f"{objective['version']} sha256={hash_file(objective_path)}; "
            f"protocol sha256={hash_file(protocol_path)}",
        )
        artifact_path = directory / f"routing_{n}.json"
        save(artifact_path, artifact.model_dump(mode="json"))
        artifacts[n] = {"path": str(artifact_path), "sha256": hash_file(artifact_path)}
        save(directory / f"samples_{n}.json", all_rows[n])
        reports = [
            cascade_analysis(all_rows[n], p, mode=mode)
            for p in portfolios
            for mode in ("none", "confidence", "selective", "always")
        ]
        checks = {}
        for tier, report in tier_reports[n].items():
            s = report["selective"]
            checks[f"{tier}_error_auroc"] = (
                s["error_auroc"] is not None
                and s["error_auroc"] >= objective["gate_min_error_auroc"]
            )
            checks[f"{tier}_confidence_gain"] = (
                s["gain"] is not None and s["gain"] >= objective["gate_min_confidence_auroc_gain"]
            )
        for r in reports:
            if r["mode"] != "selective":
                continue
            key = "+".join(r["portfolio"])
            for field, bound in [
                ("escalation_rate", "gate_max_escalation_rate"),
                ("additional_passes_per_candidate", "gate_max_additional_passes_per_candidate"),
                ("uq_to_initial_deterministic_cost", "gate_max_uq_to_deterministic_cost_ratio"),
            ]:
                checks[f"{key}_{field}"] = r[field] <= objective[bound]
            checks[f"{key}_recall"] = (
                r["recall_at_frozen_tier_thresholds"]
                >= objective["gate_min_recall_at_frozen_tier_thresholds"]
            )
        cost = float(
            np.mean(
                [
                    r["mean_uq_ms"]
                    for r in reports
                    if r["mode"] == "selective" and len(r["portfolio"]) == 1
                ]
            )
        )
        gates[n] = {"pass": all(checks.values()), "checks": checks, "mean_selective_cost_ms": cost}
        analyses[n] = {"tiers": tier_reports[n], "cascades": reports}
        save(directory / f"analysis_{n}.json", analyses[n])
    passing = [n for n, g in gates.items() if g["pass"]]
    selected = (
        min(passing, key=lambda n: (gates[n]["mean_selective_cost_ms"], n)) if passing else None
    )
    gate = {
        "pass": selected is not None,
        "budgets": gates,
        "selected_passes": selected,
        "proposed_uncertainty_policy": "selective_mc_dropout" if selected else "none",
        "proposed_escalation_policy": "cascade" if selected else "none",
        "negative_ablation_passes": None if selected else 5,
    }
    save(directory / "gate_c.json", gate)
    save(directory / "calibration_baselines.json", calibration_baselines)
    plot(directory, all_rows, analyses)
    # This is a frozen evidence inventory, not a mutable latest-file overwrite.
    source = {}
    for folder in ("src", "scripts", "tests", "configs", "docs", "references"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source[str(path.relative_to(ROOT))] = hash_file(path)
    save(directory / "source_inventory.json", source)
    inventory = {p.name: hash_file(p) for p in sorted(directory.iterdir()) if p.is_file()}
    save(
        directory / "index.json",
        {
            "schema_version": 1,
            "identifier": identifier,
            "partition": "validation",
            "candidate_count": len(samples),
            "test_evaluated": False,
            "dataset_hashes": [a.model_dump(mode="json") for a in dataset_hashes(config)],
            "artifacts": artifacts,
            "gate_c": gate,
            "files": inventory,
            "objective_sha256": hash_file(objective_path),
            "protocol_sha256": hash_file(protocol_path),
            "limitations": objective["limitations"],
        },
    )
    print(json.dumps({"index": str(directory / "index.json"), "gate_c": gate}, indent=2))


def plot(directory, all_rows, analyses):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for column, n in enumerate((5, 10, 20)):
        for row, tier in enumerate(("tiny", "medium", "large")):
            records = all_rows[n][tier]
            ax = axes[row, column]
            for correct, color in ((True, "#247a9b"), (False, "#b84040")):
                values = [r["std"] for r in records if ((r["p"] >= 0.5) == r["label"]) == correct]
                ax.hist(
                    values,
                    bins=np.linspace(0, 0.5, 21),
                    alpha=0.6,
                    color=color,
                    label=f"{'Correct' if correct else 'Incorrect'} (n={len(values)})",
                )
            ax.set(
                title=f"{tier.capitalize()} / {n} passes",
                xlabel="MC population std",
                ylabel="Validation candidates",
                yscale="symlog",
            )
            ax.legend(fontsize=8)
    fig.suptitle("Validation only: deterministic errors and MC dispersion (fit-set diagnostics)")
    fig.savefig(directory / "uncertainty_errors.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for tier in ("tiny", "medium", "large"):
        q = analyses[5]["tiers"][tier]["always"]["error_quantiles"]
        axes[0].plot([x["quantile"] for x in q], [x["error_rate"] for x in q], "o-", label=tier)
    axes[0].set(xlabel="MC std quantile (5 passes; ties kept together)", ylabel="Error rate at 0.5")
    axes[0].legend()
    for mode in ("none", "confidence", "selective", "always"):
        budgets = (5,) if mode in {"none", "confidence"} else (5, 10, 20)
        rows = [
            next(
                r
                for r in analyses[n]["cascades"]
                if r["portfolio"] == ["tiny", "medium", "large"] and r["mode"] == mode
            )
            for n in budgets
        ]
        axes[1].plot(
            [r["mean_ms"] for r in rows],
            [r["recall_at_frozen_tier_thresholds"] for r in rows],
            "o-",
            label=mode,
        )
        if len(budgets) > 1:
            for n, r in zip(budgets, rows):
                axes[1].annotate(
                    str(n),
                    (r["mean_ms"], r["recall_at_frozen_tier_thresholds"]),
                    fontsize=8,
                    xytext=(3, 6),
                    textcoords="offset points",
                )
    axes[1].legend(title="Policy (numbers = MC passes)", fontsize=8, loc="center right")
    axes[1].axhline(0.95, color="grey", linestyle="--")
    axes[1].set(
        xlabel="Measured component cost / candidate (ms)",
        ylabel="Recall at frozen tier thresholds",
        title="Three-tier portfolio",
    )
    fig.suptitle("Validation only: uncertainty/error and recall/cost diagnostics")
    fig.savefig(directory / "validation_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/uncertainty")
    parser.add_argument("--replot", type=Path, help="Existing index; no inference or fitting")
    args = parser.parse_args()
    if args.replot:
        args.output.mkdir(parents=True, exist_ok=False)
        source = args.replot.parent
        rows = {n: json.loads((source / f"samples_{n}.json").read_text()) for n in (5, 10, 20)}
        analyses = {n: json.loads((source / f"analysis_{n}.json").read_text()) for n in (5, 10, 20)}
        plot(args.output, rows, analyses)
        save(
            args.output / "index.json",
            {
                "benchmark_index_sha256": hash_file(args.replot),
                "reason": "Remove overlapping baseline labels; measurements unchanged",
                "files": {p.name: hash_file(p) for p in args.output.glob("*.png")},
            },
        )
    else:
        benchmark(args.output)
