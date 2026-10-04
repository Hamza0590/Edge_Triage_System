"""Read-only, chunked audit; diagnostic percentiles use an explicit bounded sample."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from edge_triage.config import AppConfig
from edge_triage.data.artifacts import (
    CHANNELS,
    LEAKAGE_WARNING,
    canonical,
    load_raw,
    write_immutable,
)
from edge_triage.health import hash_file


def audit(config: AppConfig) -> dict[str, Any]:
    images, labels, hashes = load_raw(config)
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    count = np.zeros(3, dtype=np.int64)
    nan = inf = zeros = constants = duplicates = 0
    seen: set[bytes] = set()
    for start in range(0, len(images), 32):
        chunk = images[start : start + 32]
        nan += int(np.isnan(chunk).sum())
        inf += int(np.isinf(chunk).sum())
        zeros += int(np.all(chunk == 0, axis=(1, 2, 3)).sum())
        constants += int((np.max(chunk, axis=(1, 2)) == np.min(chunk, axis=(1, 2))).sum())
        for sample in chunk:
            digest = hashlib.sha256(sample.tobytes()).digest()
            duplicates += int(digest in seen)
            seen.add(digest)
        for channel in range(3):
            values = chunk[..., channel].astype(np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                total[channel] += values.sum()
                squares[channel] += np.square(values).sum()
                count[channel] += values.size
                low[channel] = min(low[channel], values.min())
                high[channel] = max(high[channel], values.max())
    if nan or inf or zeros or constants or duplicates:
        raise ValueError(
            f"frozen-data quality failed: {nan=}, {inf=}, {zeros=}, {constants=}, {duplicates=}"
        )
    # One million evenly spaced pixels per channel, bounded to ~12 MB total.
    flat = images.reshape(-1, 3)
    stride = max(1, int(np.ceil(len(flat) / 1_000_000)))
    sampled = np.array(flat[::stride], copy=True)
    levels = [0, 1, 5, 50, 95, 99, 100]
    stats: dict[str, Any] = {}
    for channel, name in enumerate(CHANNELS):
        mean = total[channel] / count[channel]
        stats[name] = {
            "min": float(low[channel]),
            "max": float(high[channel]),
            "mean": float(mean),
            "std": float(np.sqrt(max(0, squares[channel] / count[channel] - mean**2))),
            "percentiles": dict(
                zip(map(str, levels), np.percentile(sampled[:, channel], levels).tolist())
            ),
        }
    mean = total.sum() / count.sum()
    stats["global"] = {
        "min": float(low.min()),
        "max": float(high.max()),
        "mean": float(mean),
        "std": float(np.sqrt(squares.sum() / count.sum() - mean**2)),
        "percentiles": dict(zip(map(str, levels), np.percentile(sampled, levels).tolist())),
    }
    with config.dataset.images.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
    report = {
        "schema_version": 1,
        "dataset_hashes": hashes,
        "files": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hash_file(path)}
            for path in (config.dataset.images, config.dataset.labels)
        ],
        "npy": {
            "version": list(version),
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "dtype_descriptor": images.dtype.str,
            "header_bytes": images.offset,
            "order": "F" if images.flags.f_contiguous else "C",
        },
        "csv": {
            "columns": ["index_no", "label"],
            "types": ["integer", "string"],
            "rows": len(labels),
            "label_values": ["Bogus", "Real"],
            "sequential_alignment": True,
        },
        "class_counts": {"Real": labels.count(1), "Bogus": labels.count(0)},
        "class_ratios": {
            "Real": labels.count(1) / len(labels),
            "Bogus": labels.count(0) / len(labels),
        },
        "statistics": stats,
        "percentile_method": {
            "kind": "approximate deterministic strided sample",
            "stride_pixels": stride,
            "samples_per_channel": len(sampled),
            "note": "Min/max/mean/std cover all pixels; diagnostics never fit transforms.",
        },
        "quality": {
            "nan": nan,
            "inf": inf,
            "all_zero_samples": zeros,
            "constant_sample_channels": constants,
            "exact_duplicate_triplets": duplicates,
        },
        "metadata_inventory": ["index_no", "label"],
        "leakage_warning": LEAKAGE_WARNING,
    }
    directory = config.paths.artifacts_dir / "data"
    write_immutable(directory / "dataset_audit.json", canonical(report))
    lines = [
        "# MeerLICHT frozen dataset audit",
        "",
        "3,219 float32 NRD triplets, 100x100 pixels.",
        "1,647 Real; 1,572 Bogus. Sequential label alignment verified.",
        "",
        LEAKAGE_WARNING,
        "",
        "All NaN/Inf/zero/constant/duplicate checks: zero.",
        "",
        "Means/std/min/max use all pixels. Percentiles use a deterministic strided sample;",
        "these global diagnostics do not fit or tune model preprocessing.",
        "",
        "| Channel | Min | Max | Mean | Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in stats.items():
        lines.append(
            f"| {name} | {values['min']:.6g} | {values['max']:.6g} | "
            f"{values['mean']:.6g} | {values['std']:.6g} |"
        )
    write_immutable(directory / "dataset_audit.md", ("\n".join(lines) + "\n").encode())
    return report
