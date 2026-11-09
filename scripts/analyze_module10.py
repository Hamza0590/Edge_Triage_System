"""Verify Module10 evidence and produce validation-only tables and figures."""

import csv
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from edge_triage.config import load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import write_immutable
from edge_triage.health import hash_file
from edge_triage.runtime.records import CandidateRecord
from edge_triage.triage.artifacts import TierFateDefinition, load_fate_artifact
from edge_triage.triage.fitting import bootstrap_fates, evaluate_fates, fit_thresholds
from edge_triage.triage.policies import LogicalByteAccounting
from edge_triage.triage.workflow import read_collection, row_evidence, save_json, verify_frozen_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/triage/module10_v2"
REPORT = OUT / "analysis"
FATES = ("transmit_all", "binary", "common_three_way", "recall_constrained")
METRICS = (
    "survival_recall",
    "classifier_recall",
    "real_transmit_recall",
    "real_compress_recall",
    "bogus_discard_rate",
    "false_discard_count",
    "false_discard_rate",
    "bytes_transmitted",
    "bytes_original",
    "byte_reduction",
)


def table(name, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_immutable(REPORT / name, stream.getvalue().encode())


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((OUT / "matrix.json").read_bytes())
    protocol = json.loads((OUT / "protocol.json").read_bytes())
    fits, replays, false_discards, distributions, points, costs = [], [], [], [], [], []
    reports, artifacts, evidence_by_name, rows_by_name = {}, {}, {}, {}
    source_hashes, registry = {}, []
    accounting = LogicalByteAccounting()
    for entry in matrix["settings"]:
        name = entry["name"]
        directory = OUT / "settings" / name
        config = load_config(entry["config"])
        assert hash_file(Path(entry["config"])) == entry["sha256"]
        rows, evidence = read_collection(directory / "collection.json", config)
        evidence_by_name[name], rows_by_name[name] = evidence, rows
        for fate in FATES:
            dest = directory / fate
            frozen = load_config(dest / "config.yaml")
            artifact = load_fate_artifact(
                dest / "threshold.json", frozen.threshold.artifact_sha256, frozen
            )
            report = json.loads((dest / "fit.json").read_bytes())
            assert report["collection_sha256"] == hash_file(directory / "collection.json")
            assert artifact.validation_records_sha256 == evidence["hashes"]["candidates.parquet"]
            metrics = evaluate_fates(rows, artifact.tiers, accounting, binary=fate == "binary")
            # Summing floats in runtime completion order may differ from canonical fit order.
            for key in metrics:
                if key == "total_end_to_end_ms":
                    assert np.isclose(metrics[key], report["metrics"][key], rtol=1e-12)
                elif key == "false_discard_review":
                    assert sorted(metrics[key], key=lambda x: x["candidate_id"]) == sorted(
                        report["metrics"][key], key=lambda x: x["candidate_id"]
                    )
                else:
                    assert metrics[key] == report["metrics"][key], (name, fate, key)
            assert (
                bootstrap_fates(
                    rows,
                    artifact.tiers,
                    accounting,
                    repeats=protocol["bootstrap"]["repeats"],
                    seed=protocol["bootstrap"]["seed"],
                    binary=fate == "binary",
                )
                == report["bootstrap"]
            )
            assert metrics["survival_recall"] >= 0.95
            reports[name, fate], artifacts[name, fate] = report, artifact
            fits.append(
                {
                    "setting": name,
                    "fate": fate,
                    "status": report["status"],
                    **{k: metrics[k] for k in METRICS},
                }
            )
            for item in metrics["false_discard_review"]:
                false_discards.append({"setting": name, "fate": fate, "phase": "fit", **item})
            for label_tier, counts in metrics["fate_distribution_label_tier"].items():
                label, tier = label_tier.split(":")
                distributions.append(
                    {
                        "setting": name,
                        "fate": fate,
                        "phase": "fit",
                        "label": label,
                        "tier": tier,
                        **counts,
                    }
                )
            for spec in artifact.tiers:
                points.append(
                    {
                        "setting": name,
                        "fate": fate,
                        "tier": spec.tier.value,
                        "discard_supported": spec.discard_supported,
                        "probability_mode": spec.probability_mode,
                        **spec.operating_point.model_dump(),
                    }
                )
            for file in ("fit.json", "threshold.json", "config.yaml"):
                source_hashes[str((dest / file).relative_to(ROOT))] = hash_file(dest / file)
        replay_path = directory / "runtime_replay.json"
        saved = json.loads(replay_path.read_bytes())
        replay_config = load_config(directory / "recall_constrained/config.yaml")
        assert verify_frozen_run(replay_config, Path(saved["run"])) == saved
        metrics = saved["metrics"]
        replays.append(
            {"setting": name, "status": saved["status"], **{k: metrics[k] for k in METRICS}}
        )
        for item in metrics["false_discard_review"]:
            false_discards.append(
                {"setting": name, "fate": "recall_constrained", "phase": "runtime_replay", **item}
            )
        for label_tier, counts in metrics["fate_distribution_label_tier"].items():
            label, tier = label_tier.split(":")
            distributions.append(
                {
                    "setting": name,
                    "fate": "recall_constrained",
                    "phase": "runtime_replay",
                    "label": label,
                    "tier": tier,
                    **counts,
                }
            )
        for phase, data in (("collection", evidence), ("runtime_replay", saved)):
            summary = data["summary"]
            manifest = json.loads((Path(data["run"]) / "manifest.json").read_bytes())
            elapsed = (
                datetime.fromisoformat(manifest["ended_utc"])
                - datetime.fromisoformat(manifest["started_utc"])
            ).total_seconds()
            costs.append(
                {
                    "setting": name,
                    "phase": phase,
                    **data["costs"],
                    "run_elapsed_including_startup_s": elapsed,
                    "run_throughput_per_s": data["candidates"] / elapsed,
                    "scheduler_selection_ms": summary["scheduler"]["selection_total_ms"],
                    "scheduler_admission_ms": summary["scheduler"]["admission_total_ms"],
                    "monitor_sampling_seconds": summary["monitor"]["sampling_seconds"],
                    "monitor_wall_fraction": summary["monitor"]["sampling_wall_fraction"],
                    "peak_active": summary["executor"]["peak_active"],
                }
            )
        for file in ("collection.json", "runtime_replay.json", "collection_config.yaml"):
            source_hashes[str((directory / file).relative_to(ROOT))] = hash_file(directory / file)
        for fate in FATES:
            dest = directory / fate
            registry.append(
                {
                    "setting": name,
                    "fate": fate,
                    "config": str(dest / "config.yaml"),
                    "config_sha256": hash_file(dest / "config.yaml"),
                    "threshold": str(dest / "threshold.json"),
                    "threshold_sha256": hash_file(dest / "threshold.json"),
                    "fit_status": "validation_feasible",
                    "runtime_replay_status": saved["status"]
                    if fate == "recall_constrained"
                    else "fixed_upstream_fate_ablation_only",
                }
            )
        print(f"VERIFIED {name}", flush=True)

    common = []
    uq = []
    for entry in matrix["settings"]:
        name = entry["name"]
        a = reports[name, "common_three_way"]["metrics"]
        b = reports[name, "recall_constrained"]["metrics"]
        common.append(
            {
                "setting": name,
                "common_bytes": a["bytes_transmitted"],
                "per_tier_bytes": b["bytes_transmitted"],
                "per_tier_bytes_saved": a["bytes_transmitted"] - b["bytes_transmitted"],
                "common_survival_recall": a["survival_recall"],
                "per_tier_survival_recall": b["survival_recall"],
            }
        )
        if name.startswith("mc5_"):
            base = name.replace("mc5_", "cal_none_", 1)
            a = reports[base, "recall_constrained"]["metrics"]
            uq.append(
                {
                    "portfolio": name.removeprefix("mc5_"),
                    "none_false_discards": a["false_discard_count"],
                    "mc5_false_discards": b["false_discard_count"],
                    "none_bytes": a["bytes_transmitted"],
                    "mc5_bytes": b["bytes_transmitted"],
                    "none_sum_e2e_ms": evidence_by_name[base]["costs"]["end_to_end_ms"],
                    "mc5_sum_e2e_ms": evidence_by_name[name]["costs"]["end_to_end_ms"],
                    "mc5_additional_passes": evidence_by_name[name]["costs"][
                        "additional_forward_passes"
                    ],
                }
            )

    name = "proposed_all_dynamic"
    config = load_config(OUT / "settings" / name / "collection_config.yaml")
    curves = []
    for target in protocol["recall_curve_targets"]:
        result = fit_thresholds(
            rows_by_name[name],
            partition="validation",
            split_path=config.split.manifest,
            split_sha256=hash_file(config.split.manifest),
            tiers=artifacts[name, "recall_constrained"].tiers,
            accounting=accounting,
            target_survival_recall=target,
            monotonic=True,
            transmit_grid=tuple(protocol["search"]["min_p_real_for_transmit"]),
            uncertainty_grid=tuple(protocol["search"]["max_uncertainty_for_discard"]),
        )
        curves.append(
            {
                "target": target,
                **{k: result.metrics[k] for k in METRICS},
                **evidence_by_name[name]["costs"],
            }
        )
    seeds, training_costs = [], []
    for seed in protocol["seed_stability"]["training_seeds"]:
        for tier in ModelTier:
            path = OUT / "seeds" / str(seed) / tier.value / "result.json"
            result = json.loads(path.read_bytes())
            for file, digest in result["run_hashes"].items():
                assert hash_file(Path(result["run"]) / file) == digest
            seed_config = load_config(Path(result["run"]) / "config.resolved.yaml")
            assert hash_file(seed_config.models.checkpoint_index) == result["family_sha256"]
            assert hash_file(seed_config.calibration.artifact_index) == result["calibration_sha256"]
            family = json.loads(seed_config.models.checkpoint_index.read_bytes())
            checkpoint = family["checkpoints"][tier.value]
            model_dir = Path(checkpoint["directory"])
            manifest_path = model_dir / "checkpoint_manifest.json"
            assert hash_file(manifest_path) == checkpoint["manifest_sha256"]
            manifest = json.loads(manifest_path.read_bytes())
            for filename, digest in manifest["file_hashes"].items():
                assert hash_file(model_dir / filename) == digest
            calibration_index = json.loads(seed_config.calibration.artifact_index.read_bytes())
            cal_entry = calibration_index["calibrations"][tier.value]
            assert hash_file(Path(cal_entry["path"])) == cal_entry["sha256"]
            calibration = json.loads(Path(cal_entry["path"]).read_bytes())
            spec = TierFateDefinition.model_validate(result["tiers"][0])
            assert spec.checkpoint_id == manifest["checkpoint_id"]
            assert spec.calibration_artifact_id == calibration["artifact_id"]
            model_metrics = json.loads((model_dir / "metrics.json").read_bytes())
            training_costs.append(
                {
                    "seed": seed,
                    "tier": tier.value,
                    "origin": "frozen_original" if seed == 20260924 else "separate_diagnostic",
                    "training_elapsed_s": model_metrics["training_elapsed_s"],
                    "best_epoch": manifest["best_epoch"],
                    "checkpoint_id": manifest["checkpoint_id"],
                }
            )
            seed_rows = [
                row_evidence(CandidateRecord.model_validate(r))
                for r in pq.read_table(Path(result["run"]) / "candidates.parquet").to_pylist()
            ]
            checked = fit_thresholds(
                seed_rows,
                partition="validation",
                split_path=seed_config.split.manifest,
                split_sha256=hash_file(seed_config.split.manifest),
                tiers=(spec,),
                accounting=accounting,
                common=True,
                target_survival_recall=0.95,
                transmit_grid=(0.5,),
                uncertainty_grid=(0.0,),
            )
            assert checked.tiers == (spec,)
            assert checked.metrics == result["metrics"]
            seeds.append(
                {
                    "seed": seed,
                    "tier": tier.value,
                    **spec.operating_point.model_dump(),
                    **{k: result["metrics"][k] for k in METRICS},
                }
            )
            source_hashes[str(path.relative_to(ROOT))] = hash_file(path)

    for filename, data in (
        ("fit_metrics.csv", fits),
        ("runtime_replays.csv", replays),
        ("false_discard_review.csv", false_discards),
        ("label_tier_fates.csv", distributions),
        ("thresholds.csv", points),
        ("costs.csv", costs),
        ("common_vs_per_tier.csv", common),
        ("uq_effect.csv", uq),
        ("recall_curve.csv", curves),
        ("seed_stability.csv", seeds),
        ("seed_training_costs.csv", training_costs),
    ):
        table(filename, data)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = [r["survival_recall"] for r in curves]
    axes[0].plot(x, [r["bytes_transmitted"] / 1e6 for r in curves], "o-", label="Threshold sweep")
    axes[0].axhline(curves[0]["bytes_original"] / 1e6, ls="--", color="gray", label="Transmit all")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].set(ylabel="Logical payload (MB)", title="Recall and transmitted bytes")
    axes[1].plot(x, [r["end_to_end_ms"] / 643 for r in curves], "o-", label="Collection (shared)")
    primary_replay = next(r for r in replays if r["setting"] == name)
    primary_cost = next(c for c in costs if c["setting"] == name and c["phase"] == "runtime_replay")
    axes[1].scatter(
        [primary_replay["survival_recall"]],
        [primary_cost["end_to_end_ms"] / 643],
        marker="D",
        color="darkorange",
        label="Frozen runtime replay",
        zorder=3,
    )
    axes[1].set_ylim(0, 1.15 * max(curves[0]["end_to_end_ms"], primary_cost["end_to_end_ms"]) / 643)
    axes[1].legend()
    axes[1].set(ylabel="Mean runtime E2E (ms, includes waits)", title="Fixed upstream execution")
    for ax in axes:
        ax.set_xlabel("Validation survival recall")
        ax.grid(alpha=0.25)
    fig.suptitle("Proposed configuration · validation threshold sweep · no codec/network")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    write_immutable(REPORT / "recall_bytes_cost.png", buffer.getvalue())
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, tier in zip(axes, ModelTier, strict=True):
        subset = [s for s in seeds if s["tier"] == tier.value]
        ax.plot(
            [str(s["seed"]) for s in subset], [s["max_p_real_for_discard"] for s in subset], "o-"
        )
        ax.set(title=tier.value.title(), xlabel="Training seed", ylim=(0, 0.5))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Calibrated discard cutoff")
    fig.suptitle("Seed sensitivity · fixed split · singleton tiers · transmit cutoff 0.5")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    write_immutable(REPORT / "seed_stability.png", buffer.getvalue())
    plt.close(fig)
    primary = reports[name, "recall_constrained"]
    statuses = {r["setting"]: r["status"] for r in replays}
    summary = {
        "upstream_configurations": len(matrix["settings"]),
        "fate_variants": len(fits),
        "collection_candidates": sum(e["candidates"] for e in evidence_by_name.values()),
        "frozen_runtime_replay_candidates": len(replays) * 643,
        "fit_feasible": len(fits),
        "replay_statuses": statuses,
        "primary_fit": primary,
        "primary_replay": next(r for r in replays if r["setting"] == name),
        "seed_diagnostics": seeds,
        "test_evaluated": False,
        "infeasible_fallback": "Use separately fitted transmit_all config; never refit on replay.",
        "cost_scope": "Named costs overlap; E2E includes waiting, not CPU attribution. "
        "Offline fate ablations share measured upstream compute. Codec CPU=0 logical.",
        "bootstrap_scope": "Fixed thresholds, IID row percentile; conditional validation only. "
        "Repeated-object dependence unknown; no generalization guarantee.",
    }
    save_json(REPORT / "summary.json", summary)
    save_json(REPORT / "artifact_registry.json", registry)
    save_json(REPORT / "verified_source_hashes.json", source_hashes)
    save_json(
        REPORT / "artifact_hashes.json",
        {
            p.name: hash_file(p)
            for p in sorted(REPORT.iterdir())
            if p.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "upstream_configurations",
                    "fate_variants",
                    "fit_feasible",
                    "primary_replay",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
