"""Supervised FP32/AMP training. Only train and validation tensors are loaded."""

from __future__ import annotations

import csv
import io
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from edge_triage.config import AppConfig
from edge_triage.data.artifacts import (
    canonical,
    frozen_split,
    load_raw,
    split_path,
    write_immutable,
)
from edge_triage.data.loaders import make_loader
from edge_triage.data.preprocessing import load_profile, profile_identity, profile_path
from edge_triage.health import hash_file
from edge_triage.models.cnn import ScalableRealBogusCNN, TierSpec, load_spec, resolve_device
from edge_triage.models.registry import CHECKPOINT_FILES, CheckpointManifest
from edge_triage.runs import RunManager


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    # FP32 reference explicitly excludes TF32 acceleration.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def binary_metrics(labels: Any, logits: Any) -> dict[str, float]:
    """Non-interpolated PR area (average precision); group tied scores together."""
    y, scores = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if (
        y.ndim != 1
        or scores.shape != y.shape
        or not len(y)
        or not np.isfinite(scores).all()
        or not np.isin(y, [0, 1]).all()
        or len(np.unique(y)) != 2
    ):
        raise ValueError("metrics require finite aligned scores and both binary classes")
    order = np.argsort(-scores, kind="stable")
    sorted_y, sorted_scores = y[order], scores[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tp = np.cumsum(sorted_y)[endpoints]
    recall = tp / y.sum()
    precision = tp / (endpoints + 1)
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    predicted = scores >= 0  # Exactly sigmoid(logit) >= 0.5; uncalibrated diagnostic.
    real_recall = float(np.sum(predicted & (y == 1)) / y.sum())
    return {
        "average_precision": ap,
        "recall_at_0_5": real_recall,
        "fnr_at_0_5": 1 - real_recall,
        "accuracy_at_0_5": float(np.mean(predicted == y)),
        "precision_at_0_5": float(np.sum(predicted & (y == 1)) / max(1, predicted.sum())),
        "bce": float(np.mean(np.logaddexp(0, scores) - y * scores)),
    }


def selection_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["average_precision"], metrics["recall_at_0_5"], -metrics["bce"])


def evaluate(model: ScalableRealBogusCNN, loader: Any, device: Any) -> dict[str, float]:
    if loader.dataset.partition != "validation":
        raise ValueError("training evaluation is restricted to validation")
    model.eval()
    labels, logits = [], []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["images"].to(device))
            if not torch.isfinite(output).all():
                raise FloatingPointError("non-finite validation logits")
            labels.extend(batch["labels"].tolist())
            logits.extend(output.cpu().tolist())
    return binary_metrics(labels, logits)


def preliminary_latency(
    model: ScalableRealBogusCNN, image: Any, repeats: int = 100
) -> dict[str, Any]:
    device = next(model.parameters()).device
    sample = image.unsqueeze(0).to(device)
    model.eval()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    measurements = []
    with torch.inference_mode():
        for _ in range(20):
            model(sample)
        synchronize()
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(sample)
            synchronize()
            measurements.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": float(np.median(measurements)),
        "p95_ms": float(np.percentile(measurements, 95)),
        "repeats": repeats,
        "device": str(device),
        "scope": "warm batch1 model-only wall time; synchronized",
    }


def overfit_diagnostic(
    spec: TierSpec, images: Any, labels: Any, device: Any, seed: int, steps: int = 100
) -> dict[str, Any]:
    """Separate debug model: dropout off/BN frozen; never used for model selection."""
    seed_everything(seed)
    model = ScalableRealBogusCNN(spec).to(device).eval()
    x, y = images.to(device), labels.to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.BCEWithLogitsLoss()
    initial = float(criterion(model(x), y).item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite overfit diagnostic")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("non-finite overfit gradient")
        optimizer.step()
    with torch.inference_mode():
        output = model(x)
        final = float(criterion(output, y).item())
        accuracy = float(((output >= 0) == y.bool()).float().mean().item())
    return {
        "initial_loss": initial,
        "final_loss": final,
        "accuracy": accuracy,
        "steps": steps,
        "examples": len(y),
        "finite_gradients": True,
        "passed": final < initial * 0.5 and accuracy >= 0.95,
        "mode": "separate debug model; dropout disabled; BatchNorm frozen",
    }


def resolve_training_data(config: AppConfig) -> tuple[AppConfig, dict[str, str]]:
    _, labels, hashes = load_raw(config)
    _, split_hash = frozen_split(config, labels)
    identity = profile_identity(config, split_hash, hashes)
    load_profile(config, identity)
    raw = config.model_dump()
    raw["split"]["manifest"] = split_path(config)
    raw["preprocessing"]["artifact"] = profile_path(config, identity)
    raw["loader"]["batch_size"] = config.training.batch_size
    return AppConfig.model_validate(raw), {
        f"{item['artifact_id']}:{item['algorithm']}": item["digest"] for item in hashes
    }


def train_tier(config: AppConfig, spec: TierSpec) -> dict[str, Any]:
    config, hashes = resolve_training_data(config)
    if (
        spec.input_channels != len(config.preprocessing.selected_channels)
        or spec.crop_size != config.preprocessing.crop_size
    ):
        raise ValueError("architecture/preprocessing shape mismatch")
    assert config.split.manifest is not None and config.preprocessing.artifact is not None
    device = resolve_device(config.training.device)
    if config.training.precision == "amp" and device.type != "cuda":
        raise ValueError("AMP training currently requires CUDA")
    torch.set_num_threads(config.training.cpu_threads)
    seed_everything(config.training.seed, config.training.deterministic)
    start = time.perf_counter()
    with RunManager(config) as run:

        def event(name: str, payload: dict[str, Any]) -> None:
            run.logger.emit(
                module="training",
                event_type=name if name.startswith("error.") else f"model.{name}",
                reason_code="TRAIN_VALIDATION_ONLY",
                message=name,
                payload={"tier": spec.tier.value, **payload},
            )

        event(
            "training.started",
            {
                "seed": config.training.seed,
                "device": str(device),
                "precision": config.training.precision,
            },
        )
        train = make_loader(config, "train", run_id=run.manifest.run_id)
        validation = make_loader(config, "validation", run_id=run.manifest.run_id)
        model = ScalableRealBogusCNN(spec).to(device)
        optimizer_cls = (
            torch.optim.AdamW if config.training.optimizer == "adamw" else torch.optim.Adam
        )
        optimizer = optimizer_cls(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        amp = config.training.precision == "amp"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        history: list[dict[str, Any]] = []
        best_metrics: dict[str, float] = {}
        best_state, best_epoch, stale = None, 0, 0
        peak_rss = psutil.Process().memory_info().rss
        try:
            for epoch in range(1, config.training.epochs + 1):
                epoch_start = time.perf_counter()
                event("epoch.started", {"epoch": epoch})
                train.dataset.set_epoch(epoch - 1)
                train.generator.manual_seed(config.training.seed + epoch - 1)
                model.train()
                total_loss, examples = 0.0, 0
                for batch in train:
                    x, y = batch["images"].to(device), batch["labels"].to(device).float()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        output = model(x)
                        loss = criterion(output, y)
                    if not torch.isfinite(output).all() or not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training logit/loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise FloatingPointError("non-finite training gradient")
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += float(loss.item()) * len(y)
                    examples += len(y)
                    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                metrics = evaluate(model, validation, device)
                improved = not best_metrics or selection_key(metrics) > selection_key(best_metrics)
                if improved:
                    best_metrics, best_epoch, stale = metrics, epoch, 0
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    stale += 1
                row = {
                    "epoch": epoch,
                    "train_loss": total_loss / examples,
                    **metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "stale_epochs": stale,
                    "best_epoch": best_epoch,
                    "elapsed_s": time.perf_counter() - epoch_start,
                }
                history.append(row)
                event("epoch.completed", row)
                print(
                    f"{spec.tier.value} epoch {epoch}: AP={metrics['average_precision']:.5f} "
                    f"recall={metrics['recall_at_0_5']:.4f} loss={metrics['bce']:.4f}",
                    flush=True,
                )
                if stale >= config.training.patience:
                    event("training.early_stopped", {"epoch": epoch, "stale_epochs": stale})
                    break
        except FloatingPointError:
            event("error.non_finite", {"finite": False})
            raise
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        repeated = evaluate(model, validation, device)
        if repeated != best_metrics:
            raise RuntimeError("best checkpoint validation failed repeatability check")
        result: dict[str, Any] = {
            "validation": best_metrics,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_examples": len(train.dataset),
            "validation_examples": len(validation.dataset),
            "selection": "max (validation average precision, recall@0.5, -BCE); earliest exact tie",
            "test_evaluated": False,
            "finite_logits_and_gradients": True,
            "validation_repeatable": True,
            "precision": config.training.precision,
            "training_elapsed_s": time.perf_counter() - start,
            "peak_host_rss_sampled_bytes": peak_rss,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None,
            "parameter_count": model.parameter_count,
            **model.operation_counts(),
            "preliminary_latency": preliminary_latency(model, validation.dataset[0][1]),
        }
        if config.training.debug_overfit:
            positions = [
                i
                for label in (0, 1)
                for i in [j for j, r in enumerate(train.dataset.rows) if r["label"] == label][:4]
            ]
            samples = [train.dataset[i] for i in positions]
            result["overfit_diagnostic"] = overfit_diagnostic(
                spec,
                torch.stack([s[1] for s in samples]),
                torch.tensor([s[2] for s in samples]),
                device,
                config.training.seed,
            )
            result["overfit_diagnostic"]["dataset_indices"] = [s[0].dataset_index for s in samples]
            event("training.overfit_diagnostic", result["overfit_diagnostic"])
            if not result["overfit_diagnostic"]["passed"]:
                raise RuntimeError("debug overfit sanity gate failed")
        directory = config.paths.artifacts_dir / "models" / spec.tier.value / run.manifest.run_id
        directory.mkdir(parents=True, exist_ok=False)
        buffer = io.BytesIO()
        torch.save(best_state, buffer)
        write_immutable(directory / "model_state.pt", buffer.getvalue())
        result["checkpoint_bytes"] = (directory / "model_state.pt").stat().st_size
        for name, value in (
            ("model_config.yaml", spec.model_dump(mode="json")),
            ("training_config.yaml", config.training.model_dump(mode="json")),
        ):
            write_immutable(directory / name, yaml.safe_dump(value, sort_keys=True).encode())
        for name, path in (
            ("preprocessing_reference.json", config.preprocessing.artifact),
            ("split_reference.json", config.split.manifest),
        ):
            write_immutable(
                directory / name, canonical({"path": str(path), "sha256": hash_file(path)})
            )
        write_immutable(directory / "metrics.json", canonical(result))
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        write_immutable(directory / "history.csv", csv_buffer.getvalue().encode())
        checkpoint_id = f"{spec.tier.value}:{spec.architecture_version}:{run.manifest.run_id}"
        manifest = CheckpointManifest(
            checkpoint_id=checkpoint_id,
            tier=spec.tier,
            architecture_version=spec.architecture_version,
            parameter_count=model.parameter_count,
            dataset_hashes=hashes,
            split_sha256=hash_file(config.split.manifest),
            preprocessing_sha256=hash_file(config.preprocessing.artifact),
            seed=config.training.seed,
            git_commit=run.manifest.git_commit,
            best_epoch=best_epoch,
            selection_metric="validation_average_precision",
            file_hashes={name: hash_file(directory / name) for name in sorted(CHECKPOINT_FILES)},
        )
        manifest_path = directory / "checkpoint_manifest.json"
        write_immutable(manifest_path, canonical(manifest.model_dump(mode="json")))
        run.register_checkpoint(manifest_path, checkpoint_id)
        entry = {
            "checkpoint_id": checkpoint_id,
            "directory": str(directory),
            "manifest_sha256": hash_file(manifest_path),
            "run_id": run.manifest.run_id,
        }
        event("checkpoint.saved", entry)
        event(
            "training.completed",
            {
                "best_epoch": best_epoch,
                "metrics": best_metrics,
                "elapsed_s": time.perf_counter() - start,
                "peak_host_rss_sampled_bytes": peak_rss,
                "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            },
        )
    return entry


def train_family(config: AppConfig) -> Path:
    entries = {}
    for tier in config.models.tier_portfolio:
        spec = load_spec(config.paths.project_root / "configs" / "models" / f"{tier.value}_v1.yaml")
        entries[tier.value] = train_tier(config, spec)
    index = config.paths.artifacts_dir / "models" / f"family_{entries[tier.value]['run_id']}.json"
    write_immutable(index, canonical({"schema_version": 1, "checkpoints": entries}))
    return index
