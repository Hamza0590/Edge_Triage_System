"""Separate predeclared seed diagnostics; never replace the original model family."""

import contextlib
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch

from edge_triage.calibration.metrics import metrics
from edge_triage.calibration.temperature import CalibrationArtifact, fit_temperature
from edge_triage.calibration.workflow import validation_predictions
from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest, ModelRegistry, read_index
from edge_triage.models.training import resolve_training_data, train_family
from edge_triage.runs import RunManager
from edge_triage.runtime.components import DatasetSource
from edge_triage.runtime.factory import assemble, persist_thresholds
from edge_triage.runtime.pipeline import Pipeline
from edge_triage.runtime.records import CandidateRecord, ParquetRecordWriter
from edge_triage.triage.fitting import fit_thresholds
from edge_triage.triage.policies import LogicalByteAccounting
from edge_triage.triage.workflow import row_evidence, save_json, tier_definitions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/triage/module10_v2/seeds"


def calibrate(config, output):
    config, hashes = resolve_training_data(config)
    family = read_index(config.models.checkpoint_index)
    registry = ModelRegistry(
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        dataset_hashes=hashes,
        device=config.training.device,
    )
    entries = {}
    with RunManager(config) as run:
        for tier in ModelTier:
            entry = family["checkpoints"][tier.value]
            directory = Path(entry["directory"])
            registry.register(tier, entry["checkpoint_id"], directory, entry["manifest_sha256"])
            manifest_path = directory / "checkpoint_manifest.json"
            manifest = CheckpointManifest.model_validate_json(manifest_path.read_bytes())
            run.register_checkpoint(manifest_path, entry["checkpoint_id"])
            rows = validation_predictions(
                config, registry, tier, entry["checkpoint_id"], run.manifest.run_id
            )
            y, z = [r["label"] for r in rows], [r["logit"] for r in rows]
            fitting = fit_temperature(y, z, partition="validation")
            before, after = metrics(y, z), metrics(y, z, temperature=fitting["temperature"])
            prediction_path = output / f"{tier.value}_predictions.json"
            save_json(prediction_path, rows)
            artifact = CalibrationArtifact(
                artifact_id=f"module10_seed{manifest.seed}:{tier.value}",
                checkpoint_id=manifest.checkpoint_id,
                checkpoint_manifest_sha256=entry["manifest_sha256"],
                tier=tier,
                seed=manifest.seed,
                split_sha256=manifest.split_sha256,
                preprocessing_sha256=manifest.preprocessing_sha256,
                validation_rows_sha256=hashlib.sha256(
                    canonical(
                        [{"candidate_id": r["candidate_id"], "label": r["label"]} for r in rows]
                    )
                ).hexdigest(),
                prediction_sha256=hash_file(prediction_path),
                config_sha256=config.config_hash,
                code_sha256=hash_file(Path(__file__)),
                temperature=fitting["temperature"],
                recommended_mode="calibrated"
                if all(after[k] <= before[k] + 1e-12 for k in ("nll", "brier", "ece"))
                else "uncalibrated",
                selection_criterion=(
                    "Inherited NLL/Brier/ECE rule; diagnostics explicitly force calibrated mode"
                ),
                fitting=fitting,
                before=before,
                after=after,
            )
            path = output / f"{tier.value}_temperature.json"
            save_json(path, artifact.model_dump(mode="json"))
            entries[tier.value] = {
                "path": str(path),
                "sha256": hash_file(path),
                "predictions": str(prediction_path),
                "predictions_sha256": hash_file(prediction_path),
            }
    save_json(output / "index.json", {"schema_version": 1, "calibrations": entries})
    return output / "index.json"


def diagnostic(config, directory):
    torch.set_num_threads(1)
    config, _ = resolve_training_data(config)
    with RunManager(config) as run:
        parts = assemble(config, run.logger)
        persist_thresholds(run, parts.thresholds)
        writer = ParquetRecordWriter(
            run.directory / "candidates.parquet", purpose=config.experiment.purpose
        )
        completed, failed = Pipeline(config, parts, run.logger, writer).run(
            DatasetSource(config, run.manifest.run_id, 643)
        )
        if completed != 643 or failed:
            raise RuntimeError("seed validation collection incomplete")
    records = [
        CandidateRecord.model_validate(r)
        for r in pq.read_table(run.directory / "candidates.parquet").to_pylist()
    ]
    rows = [row_evidence(r) for r in records]
    result = fit_thresholds(
        rows,
        partition="validation",
        split_path=config.split.manifest,
        split_sha256=hash_file(config.split.manifest),
        tiers=tier_definitions(config),
        accounting=LogicalByteAccounting(),
        common=True,
        target_survival_recall=0.95,
        transmit_grid=(0.5,),
        uncertainty_grid=(0.0,),
    )
    save_json(
        directory / "result.json",
        {
            "seed": config.training.seed,
            "tier": config.models.tier_portfolio[0].value,
            "metrics": result.metrics,
            "tiers": [s.model_dump(mode="json") for s in result.tiers],
            "run": str(run.directory),
            "family_sha256": hash_file(config.models.checkpoint_index),
            "calibration_sha256": hash_file(config.calibration.artifact_index),
            "run_hashes": {
                name: hash_file(run.directory / name)
                for name in (
                    "manifest.json",
                    "config.resolved.yaml",
                    "events.jsonl",
                    "candidates.parquet",
                )
            },
            "test_evaluated": False,
        },
    )


def main():
    base, _ = resolve_training_data(load_config(ROOT / "configs/training/family_v1.yaml"))
    original = load_config(ROOT / "configs/experiments/module09_proposed_v2.yaml")
    for seed in (20260924, 20260925, 20260926):
        directory = OUT / str(seed)
        directory.mkdir(parents=True, exist_ok=True)
        pointer = directory / "inputs.json"
        if pointer.exists():
            inputs = json.loads(pointer.read_bytes())
        elif seed == 20260924:
            inputs = {
                "family": str(original.models.checkpoint_index),
                "calibration": str(original.calibration.artifact_index),
            }
            save_json(pointer, inputs)
        else:
            raw = base.model_dump()
            raw["paths"]["artifacts_dir"] = directory / "new_artifacts"
            raw["training"]["seed"] = seed
            raw["experiment"].update(name=f"module10_seed{seed}", seed=seed)
            training = AppConfig.model_validate(raw)
            write_immutable(directory / "training_config.yaml", training.resolved_yaml().encode())
            print(f"TRAIN seed{seed} START", flush=True)
            with (
                open(os.devnull, "w") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                family = train_family(training)
                raw["models"]["checkpoint_index"] = family
                calibration = calibrate(AppConfig.model_validate(raw), directory / "calibration")
            inputs = {"family": str(family), "calibration": str(calibration)}
            save_json(pointer, inputs)
            print(f"TRAIN seed{seed} DONE", flush=True)
        for tier in ModelTier:
            dest = directory / tier.value
            if (dest / "result.json").exists():
                continue
            raw = original.model_dump()
            raw["models"].update(checkpoint_index=Path(inputs["family"]), tier_portfolio=(tier,))
            raw["calibration"].update(
                artifact_index=Path(inputs["calibration"]), probability_mode="calibrated"
            )
            raw["training"].update(seed=seed, device="cpu", cpu_threads=1)
            raw["tier_selection"].update(policy="fixed", fixed_tier=tier)
            raw["admission"].update(policy="sequential", execution_mode="sequential")
            raw["pipeline"].update(mode="reference", partition="validation")
            raw["workload"]["trace"] = None
            raw["fate"]["policy"] = "transmit_all"
            raw["threshold"].update(policy="common", artifact=None, artifact_sha256=None)
            raw["experiment"].update(
                name=f"seed{seed}_{tier.value}", purpose="VALIDATION_FATE_STUDY", seed=seed
            )
            print(f"DIAGNOSTIC seed{seed} {tier.value} START", flush=True)
            with (
                open(os.devnull, "w") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                diagnostic(AppConfig.model_validate(raw), dest)
            print(f"DIAGNOSTIC seed{seed} {tier.value} DONE", flush=True)


if __name__ == "__main__":
    main()
