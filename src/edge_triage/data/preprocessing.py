"""Versioned, training-only normalization and deterministic visual QA."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from edge_triage.config import AppConfig
from edge_triage.data.artifacts import (
    CHANNELS,
    LEAKAGE_WARNING,
    PARTITIONS,
    canonical,
    frozen_split,
    load_raw,
    split_path,
    write_immutable,
)
from edge_triage.health import hash_file


def crop_chw(image: Any, size: int, channels: tuple[str, ...]) -> Any:
    if image.ndim != 3 or image.shape[2] != 3 or not 0 < size <= min(image.shape[:2]):
        raise ValueError("invalid NRD image or crop size")
    if not channels or len(set(channels)) != len(channels) or not set(channels) <= set(CHANNELS):
        raise ValueError("invalid channel selection")
    y, x = (image.shape[0] - size) // 2, (image.shape[1] - size) // 2
    cropped = image[y : y + size, x : x + size, :]
    return np.ascontiguousarray(
        cropped[..., [CHANNELS.index(c) for c in channels]].transpose(2, 0, 1), dtype=np.float32
    )


def fit_statistics(
    images: Any, rows: list[dict[str, Any]], size: int, channels: tuple[str, ...]
) -> dict[str, Any]:
    total, squares = np.zeros(len(channels)), np.zeros(len(channels))
    indices = [row["index_no"] for row in rows if row["split"] == "train"]
    if not indices:
        raise ValueError("training partition is empty")
    for index in indices:
        crop = crop_chw(images[index], size, channels).astype(np.float64)
        if not np.isfinite(crop).all():
            raise ValueError("non-finite training pixels")
        total += crop.sum(axis=(1, 2))
        squares += np.square(crop).sum(axis=(1, 2))
    count = len(indices) * size * size
    mean = total / count
    std = np.sqrt(np.maximum(0, squares / count - mean**2))
    if np.any(std <= 0):
        raise ValueError("cannot standardize a constant training channel")
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_partition": "train",
        "fit_rows": len(indices),
        "pixels_per_channel": count,
        "train_indices_sha256": hashlib.sha256(canonical(indices)).hexdigest(),
    }


def profile_identity(
    config: AppConfig, split_hash: str, hashes: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": "nrd_center_crop_train_standardize_v1",
        "crop_size": config.preprocessing.crop_size,
        "channels": list(config.preprocessing.selected_channels),
        "dtype": "float32",
        "horizontal_flip": config.preprocessing.horizontal_flip,
        "vertical_flip": config.preprocessing.vertical_flip,
        "augmentation_partition": "train",
        "clipping": None,
        "split_sha256": split_hash,
        "dataset_hashes": hashes,
    }


def profile_path(config: AppConfig, identity: dict[str, Any]) -> Path:
    version = hashlib.sha256(canonical(identity)).hexdigest()[:16]
    return config.preprocessing.artifact or (
        config.paths.artifacts_dir / "data" / f"preprocessing_v1_{version}.json"
    )


def load_profile(config: AppConfig, identity: dict[str, Any]) -> dict[str, Any]:
    path = profile_path(config, identity)
    profile: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if any(profile.get(key) != value for key, value in identity.items()):
        raise ValueError("preprocessing identity does not match dataset/split/settings")
    if hash_file(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("preprocessing artifact checksum mismatch")
    means, stds = np.asarray(profile["mean"]), np.asarray(profile["std"])
    if (
        means.shape != (len(identity["channels"]),)
        or stds.shape != means.shape
        or not np.isfinite(means).all()
        or not np.isfinite(stds).all()
        or np.any(stds <= 0)
        or profile["fit_partition"] != "train"
    ):
        raise ValueError("invalid training normalization statistics")
    return profile


def contact_sheet(images: Any, rows: list[dict[str, Any]], destination: Path) -> list[int]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    selected = [
        row
        for label in (1, 0)
        for row in [r for r in rows if r["split"] == "train" and r["label"] == label][:3]
    ]
    figure, axes = plt.subplots(6, 3, figsize=(8, 13))
    for row_index, row in enumerate(selected):
        for channel, name in enumerate(CHANNELS):
            pixels = images[row["index_no"], ..., channel]
            lo, hi = np.percentile(pixels, [1, 99])
            axes[row_index, channel].imshow(pixels, cmap="gray", vmin=lo, vmax=hi)
            label = "Real" if row["label"] else "Bogus"
            axes[row_index, channel].set_title(f"{label} #{row['index_no']} - {name}", fontsize=10)
            axes[row_index, channel].axis("off")
    figure.suptitle(
        "Training examples: display scaling is not model preprocessing\n"
        "Per-panel 1st-99th percentile grayscale stretch",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120, metadata={"Software": "edge-triage"})
    plt.close(figure)
    write_immutable(destination, buffer.getvalue())
    return [int(row["index_no"]) for row in selected]


def prepare(config: AppConfig) -> dict[str, Any]:
    if config.preprocessing.normalization != "train_standardize":
        raise ValueError(
            "Module 02 baseline requires train_standardize; use configs/data/meerlicht.yaml"
        )
    images, labels, hashes = load_raw(config)
    rows, split_hash = frozen_split(config, labels, create=True)
    identity = profile_identity(config, split_hash, hashes)
    profile = {
        **identity,
        **fit_statistics(
            images, rows, config.preprocessing.crop_size, config.preprocessing.selected_channels
        ),
    }
    path = profile_path(config, identity)
    write_immutable(path, canonical(profile))
    write_immutable(path.with_suffix(".sha256"), (hash_file(path) + "\n").encode())
    directory = config.paths.artifacts_dir / "data"
    counts = {
        split: {
            "Bogus": sum(r["split"] == split and r["label"] == 0 for r in rows),
            "Real": sum(r["split"] == split and r["label"] == 1 for r in rows),
        }
        for split in PARTITIONS
    }
    qa = directory / f"contact_sheet_seed{config.split.seed}_v1.png"
    selected = contact_sheet(images, rows, qa)
    summary = {
        "schema_version": 1,
        "seed": config.split.seed,
        "algorithm": "PCG64 per-class shuffle, largest remainder v1",
        "split_path": str(split_path(config)),
        "split_sha256": split_hash,
        "dataset_hashes": hashes,
        "counts": counts,
        "preprocessing_path": str(path),
        "preprocessing_sha256": hash_file(path),
        "visualization_path": str(qa),
        "visualization_sha256": hash_file(qa),
        "visualization_indices": selected,
        "leakage_warning": LEAKAGE_WARNING,
    }
    write_immutable(
        directory / f"data_manifest_seed{config.split.seed}_{path.stem}.json", canonical(summary)
    )
    return summary
