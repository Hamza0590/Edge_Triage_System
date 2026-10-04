"""Module 05 orchestration. No test tensors, test predictions, or test metrics."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from edge_triage.calibration.metrics import bootstrap, metrics, probabilities, recall_threshold
from edge_triage.calibration.profiling import MeasuredProfile, ProfileBundle
from edge_triage.calibration.temperature import CalibrationArtifact, fit_temperature
from edge_triage.config import AppConfig
from edge_triage.contracts import ModelTier
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.data.loaders import make_loader
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest, ModelRegistry, read_index
from edge_triage.models.training import resolve_training_data, seed_everything
from edge_triage.runs import RunManager


def code_identity(root: Path) -> tuple[str, dict[str, str]]:
    paths = sorted((root / "src/edge_triage").rglob("*.py"))
    paths += [root / "scripts/profile_pipeline_overhead.py", root / "tests/fixtures/dummy.py"]
    hashes = {str(p.relative_to(root)): hash_file(p) for p in paths}
    return hashlib.sha256(canonical(hashes)).hexdigest(), hashes


def validation_predictions(
    config: AppConfig, registry: ModelRegistry, tier: ModelTier, checkpoint_id: str, run_id: str
) -> list[dict[str, Any]]:
    if config.evaluation.partition != "validation" or config.pipeline.partition != "validation":
        raise ValueError("Module 05 forbids non-validation predictions")
    loader = make_loader(config, "validation", run_id=run_id)
    model, _ = registry.load(tier, checkpoint_id)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["images"].to(registry.device)).cpu().numpy()
            for candidate, label, logit in zip(
                batch["candidates"], batch["labels"], logits, strict=True
            ):
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "label": int(label),
                        "tier": tier.value,
                        "checkpoint_id": checkpoint_id,
                        "logit": float(logit),
                    }
                )
    del model, loader
    return rows


def evaluate_saved(path: Path, config: AppConfig, temperature: float) -> dict[str, Any]:
    """Single evaluator consumes exact persisted predictions for every diagnostic."""
    rows = json.loads(path.read_text())
    y, z = [r["label"] for r in rows], [r["logit"] for r in rows]
    threshold = recall_threshold(y, z, config.fate.target_survival_recall)
    result: dict[str, Any] = {"equal_recall_raw_logit_threshold": threshold}
    for name, t in (("uncalibrated", 1), ("calibrated", temperature)):
        result[name] = {
            "at_0_5": metrics(y, z, temperature=t, bins=config.calibration.bins),
            "at_target_recall": metrics(
                y, z, temperature=t, threshold_logit=threshold, bins=config.calibration.bins
            ),
            "confidence_intervals_at_target": bootstrap(
                y,
                z,
                temperature=t,
                threshold=threshold,
                bins=config.calibration.bins,
                repeats=config.evaluation.bootstrap_repeats,
                seed=config.calibration.seed,
            ),
        }
    return result


def gate_b(
    quality: dict[str, Any], profiles: list[MeasuredProfile], config: AppConfig
) -> dict[str, Any]:
    tiny = quality["tiny"]["uncalibrated"]["at_target_recall"]
    large = quality["large"]["uncalibrated"]["at_target_recall"]
    improvement = large["average_precision"] - tiny["average_precision"]
    quality_pass = (
        improvement >= config.evaluation.gate_ap_improvement
        and large["precision"] >= tiny["precision"]
        and min(tiny["recall"], large["recall"]) >= config.fate.target_survival_recall
    )
    costs = []
    for device in sorted({p.device for p in profiles}):
        pair = {p.tier: p for p in profiles if p.device == device and p.batch_size == 1}
        t, large_p = pair[ModelTier.TINY], pair[ModelTier.LARGE]
        latency_ratio = large_p.latency_p50_ms / t.latency_p50_ms
        memory_ratio = (
            large_p.cuda_allocated_peak_bytes / t.cuda_allocated_peak_bytes
            if large_p.cuda_allocated_peak_bytes and t.cuda_allocated_peak_bytes
            else None
        )
        passed = (
            t.stable
            and large_p.stable
            and (
                latency_ratio >= config.evaluation.gate_cost_ratio
                or (memory_ratio is not None and memory_ratio >= config.evaluation.gate_cost_ratio)
            )
        )
        costs.append(
            {
                "device": device,
                "batch_size": 1,
                "latency_large_over_tiny": latency_ratio,
                "cuda_allocated_large_over_tiny": memory_ratio,
                "stable_pair": t.stable and large_p.stable,
                "pass": passed,
            }
        )
    cost_pass = any(row["pass"] for row in costs)
    return {
        "pass": quality_pass and cost_pass,
        "cost_pass": cost_pass,
        "quality_pass": quality_pass,
        "cost_evidence": costs,
        "ap_improvement_large_minus_tiny": improvement,
        "quality_rule": "AP gain >= configured delta AND Large precision >= Tiny at target recall",
        "cost_rule": "stable batch1 E2E median latency OR CUDA peak allocation ratio >= configured",
        "config": config.evaluation.model_dump(mode="json"),
        "limitation": "validation point estimates; no statistical superiority or scheduling claim",
    }


def run_module05(config: AppConfig) -> Path:
    if config.pipeline.partition != "validation":
        raise ValueError("Module 05 requires validation partition")
    if config.models.tier_portfolio != tuple(ModelTier):
        raise ValueError("Module 05 requires all three tiers")
    if config.calibration.method != "temperature_scaling":
        raise ValueError("Module 05 requires temperature scaling")
    config, hashes = resolve_training_data(config)
    assert (
        config.split.manifest and config.preprocessing.artifact and config.models.checkpoint_index
    )
    index = read_index(config.models.checkpoint_index)
    code_sha, code_hashes = code_identity(config.paths.project_root)
    torch.set_num_threads(config.training.cpu_threads)
    seed_everything(config.experiment.seed)
    registry = ModelRegistry(
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        dataset_hashes=hashes,
        device=config.training.device,
    )
    with RunManager(config) as run:
        calibration_dir = config.paths.artifacts_dir / "calibration" / run.manifest.run_id
        profile_dir = config.paths.artifacts_dir / "profiles" / run.manifest.run_id
        calibration_dir.mkdir(parents=True, exist_ok=False)
        profile_dir.mkdir(parents=True, exist_ok=False)
        write_immutable(run.directory / "artifacts/code_hashes.json", canonical(code_hashes))
        # Config (including Gate B criteria) is persisted by RunManager before measurements.
        quality, calibration_entries = {}, {}
        for tier in ModelTier:
            entry = index["checkpoints"][tier.value]
            directory = Path(entry["directory"])
            registry.register(tier, entry["checkpoint_id"], directory, entry["manifest_sha256"])
            manifest_path = directory / "checkpoint_manifest.json"
            manifest = CheckpointManifest.model_validate_json(manifest_path.read_bytes())
            run.register_checkpoint(manifest_path, entry["checkpoint_id"])
            rows = validation_predictions(
                config, registry, tier, entry["checkpoint_id"], run.manifest.run_id
            )
            y, z = [r["label"] for r in rows], [r["logit"] for r in rows]
            fitting = fit_temperature(
                y,
                z,
                partition=config.evaluation.partition,
                minimum=config.evaluation.temperature_min,
                maximum=config.evaluation.temperature_max,
            )
            temperature = fitting["temperature"]
            for row, raw_p, calibrated_p in zip(
                rows, probabilities(z), probabilities(z, temperature), strict=True
            ):
                row.update(
                    seed=manifest.seed,
                    p_real_uncalibrated=float(raw_p),
                    p_real_calibrated=float(calibrated_p),
                )
            predictions = calibration_dir / f"{tier.value}_predictions.json"
            write_immutable(predictions, canonical(rows))
            results = evaluate_saved(predictions, config, temperature)
            quality[tier.value] = results
            before, after = results["uncalibrated"]["at_0_5"], results["calibrated"]["at_0_5"]
            accepted = all(after[k] <= before[k] + 1e-12 for k in ("nll", "brier", "ece"))
            artifact = CalibrationArtifact(
                artifact_id=f"temperature:{run.manifest.run_id}:{tier.value}",
                checkpoint_id=manifest.checkpoint_id,
                tier=tier,
                seed=manifest.seed,
                checkpoint_manifest_sha256=entry["manifest_sha256"],
                split_sha256=manifest.split_sha256,
                preprocessing_sha256=manifest.preprocessing_sha256,
                validation_rows_sha256=hashlib.sha256(
                    canonical(
                        [{"candidate_id": r["candidate_id"], "label": r["label"]} for r in rows]
                    )
                ).hexdigest(),
                prediction_sha256=hash_file(predictions),
                config_sha256=config.config_hash,
                code_sha256=code_sha,
                temperature=temperature,
                recommended_mode="calibrated" if accepted else "uncalibrated",
                selection_criterion="all of NLL, Brier, binary classwise ECE non-worsening (1e-12)",
                fitting=fitting,
                before=before,
                after=after,
            )
            path = calibration_dir / f"{tier.value}_temperature.json"
            write_immutable(path, canonical(artifact.model_dump(mode="json")))
            calibration_entries[tier.value] = {
                "path": str(path),
                "sha256": hash_file(path),
                "predictions": str(predictions),
                "predictions_sha256": hash_file(predictions),
            }
            run.logger.emit(
                module="calibration",
                event_type="model.calibration.completed",
                reason_code="VALIDATION_ONLY",
                payload={
                    "tier": tier.value,
                    "temperature": temperature,
                    "mode": artifact.recommended_mode,
                },
            )
        calibration_index = calibration_dir / "index.json"
        write_immutable(
            calibration_index, canonical({"schema_version": 1, "calibrations": calibration_entries})
        )
        metrics_path = calibration_dir / "metrics.json"
        write_immutable(metrics_path, canonical(quality))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        profiles: list[MeasuredProfile] = []
        worker_metadata = []
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        jobs = [(tier, device) for tier in ModelTier for device in devices]
        np.random.default_rng(config.experiment.seed).shuffle(jobs)
        for tier, device in jobs:
            output = profile_dir / f"{tier.value}_{device}.json"
            run.logger.emit(
                module="profiling",
                event_type="experiment.profile.started",
                reason_code="ISOLATED_WORKER",
                payload={"tier": tier.value, "device": device},
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "edge_triage.calibration.profile_worker",
                    "--config",
                    str(run.directory / "config.resolved.yaml"),
                    "--tier",
                    tier.value,
                    "--device",
                    device,
                    "--output",
                    str(output),
                ],
                check=True,
                timeout=600,
                cwd=config.paths.project_root,
            )
            worker = json.loads(output.read_text())
            profiles.extend(MeasuredProfile.model_validate(row) for row in worker.pop("profiles"))
            worker_metadata.append(
                {
                    "tier": tier.value,
                    "device": device,
                    "path": str(output),
                    "sha256": hash_file(output),
                    **worker,
                }
            )
        overhead_path = profile_dir / "overhead.json"
        subprocess.run(
            [
                sys.executable,
                str(config.paths.project_root / "scripts/profile_pipeline_overhead.py"),
                "--output",
                str(overhead_path),
                "--count",
                str(config.profiling.overhead_candidates),
            ],
            check=True,
            timeout=180,
            cwd=config.paths.project_root,
        )
        bundle = ProfileBundle(
            config_sha256=config.config_hash,
            code_sha256=code_sha,
            family_index_sha256=hash_file(config.models.checkpoint_index),
            profiles=tuple(profiles),
            environment={"workers": worker_metadata, "cuda_supported": torch.cuda.is_available()},
            methodology={
                "input": "seeded synthetic NHWC; frozen crop/standardization; no holdouts",
                "timing": "perf_counter_ns; CUDA sync before/after; warmup per component/round",
                "end_to_end": "raw crop+normalize+stack+blocking H2D+forward+sigmoid+host output",
                "transfer": "pageable CPU input, blocking .to; CPU measures no-op .to",
                "forward": "preloaded input; excludes transfer/postprocessing",
                "components": "independent timings include sync/allocator costs; not additive",
                "memory": "CUDA allocator peak includes model/input; CPU sampled process RSS",
                "load": "verified registry incl hashes, state load, placement; outside inference",
                "latency": "batch completion; amortized service cost excludes batch formation wait",
                "controls": "FP32, TF32 off, deterministic, cuDNN benchmark off; fixed threads",
                "isolation": "one worker at a time; desktop activity uncontrolled; state snapshots",
                "stability": f"round E2E median max/min <= {config.profiling.stability_ratio}",
            },
            overhead={
                "path": str(overhead_path),
                "sha256": hash_file(overhead_path),
                **json.loads(overhead_path.read_text()),
            },
        )
        profile_path = profile_dir / "model_profiles.json"
        write_immutable(profile_path, canonical(bundle.model_dump(mode="json")))
        gate = gate_b(quality, profiles, config)
        gate_path = profile_dir / "gate_b.json"
        write_immutable(gate_path, canonical(gate))
        summary = {
            "schema_version": 1,
            "test_evaluated": False,
            "run_id": run.manifest.run_id,
            "calibration_index": str(calibration_index),
            "profiles": str(profile_path),
            "gate_b": gate,
            "stable_profiles": all(p.stable for p in profiles),
            "artifacts": {
                str(p): hash_file(p)
                for p in (calibration_index, metrics_path, profile_path, gate_path, overhead_path)
            },
            "limitations": [
                "validation reuse for checkpoint selection, T and recall threshold",
                "no generalization guarantee; one seed per tier; no grouped IDs",
            ],
        }
        summary_path = run.directory / "metrics/module05_summary.json"
        write_immutable(summary_path, canonical(summary))
        run.logger.emit(
            module="evaluation",
            event_type="experiment.module05.completed",
            reason_code="ARTIFACTS_FROZEN",
            payload={"gate_b_pass": gate["pass"], "summary": str(summary_path)},
        )
    return summary_path
