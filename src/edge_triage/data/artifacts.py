"""Canonical immutable data artifacts and deterministic stratified partitioning."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from edge_triage.config import AppConfig
from edge_triage.health import dataset_hashes

CHANNELS = ("new", "reference", "difference")
PARTITIONS = ("train", "validation", "test")
LEAKAGE_WARNING = "Repeated-object leakage cannot be checked: no object/time/coordinate metadata."


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite different artifact: {path}; use a new version"
            )
        return
    with path.open("xb") as stream:
        stream.write(content)


def load_raw(config: AppConfig) -> tuple[Any, list[int], list[dict[str, Any]]]:
    hashes = [item.model_dump(mode="json") for item in dataset_hashes(config)]
    images = np.load(config.dataset.images, mmap_mode="r", allow_pickle=False)
    if images.shape != (3219, 100, 100, 3) or images.dtype != np.dtype("float32"):
        raise ValueError("expected float32 NHWC shape (3219,100,100,3)")
    if tuple(config.dataset.channel_order) != CHANNELS:
        raise ValueError("source channel order must be new,reference,difference")
    with config.dataset.labels.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["index_no", "label"]:
            raise ValueError("expected CSV columns index_no,label")
        rows = list(reader)
    if [row["index_no"] for row in rows] != [str(i) for i in range(len(images))]:
        raise ValueError("image/label sequential index alignment failed")
    if {row["label"] for row in rows} != {"Real", "Bogus"}:
        raise ValueError("unexpected label values")
    labels = [int(row["label"] == "Real") for row in rows]
    if labels.count(1) != 1647 or labels.count(0) != 1572:
        raise ValueError("expected 1647 Real and 1572 Bogus")
    return images, labels, hashes


def partition_rows(
    labels: list[int], seed: int, fractions: tuple[float, ...]
) -> list[dict[str, Any]]:
    """PCG64 shuffles per class; largest remainders allocate integer counts."""
    rng = np.random.Generator(np.random.PCG64(seed))
    assignments: dict[int, str] = {}
    for label in (0, 1):
        indices = rng.permutation([i for i, value in enumerate(labels) if value == label])
        ideal = np.array(fractions) * len(indices)
        counts = np.floor(ideal).astype(int)
        for index in np.argsort(-(ideal - counts), kind="stable")[: len(indices) - counts.sum()]:
            counts[index] += 1
        offset = 0
        for split, count in zip(PARTITIONS, counts, strict=True):
            for index in indices[offset : offset + count]:
                assignments[int(index)] = split
            offset += count
    rows = [
        {
            "candidate_id": f"meerlicht-{i:06d}",
            "index_no": i,
            "label": label,
            "split": assignments[i],
        }
        for i, label in enumerate(labels)
    ]
    validate_rows(rows, labels, fractions)
    return rows


def validate_rows(
    rows: list[dict[str, Any]], labels: list[int], fractions: tuple[float, ...]
) -> None:
    if [row["index_no"] for row in rows] != list(range(len(labels))):
        raise ValueError("split coverage/order/exclusivity failed")
    for index, row in enumerate(rows):
        if (
            row["label"] != labels[index]
            or row["candidate_id"] != f"meerlicht-{index:06d}"
            or row["split"] not in PARTITIONS
        ):
            raise ValueError("split identity or label mismatch")
    for split, fraction in zip(PARTITIONS, fractions, strict=True):
        values = [row["label"] for row in rows if row["split"] == split]
        if set(values) != {0, 1} or abs(len(values) - len(labels) * fraction) > 2:
            raise ValueError("split size/classes failed")
        if abs(sum(values) / len(values) - sum(labels) / len(labels)) > 0.01:
            raise ValueError("split class stratification failed")


def split_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=["candidate_id", "index_no", "label", "split"], lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def fractions(config: AppConfig) -> tuple[float, ...]:
    return (
        config.split.train_fraction,
        config.split.validation_fraction,
        config.split.test_fraction,
    )


def split_path(config: AppConfig) -> Path:
    return config.split.manifest or (
        config.paths.artifacts_dir / "data" / f"split_seed{config.split.seed}_v1.csv"
    )


def frozen_split(
    config: AppConfig, labels: list[int], *, create: bool = False
) -> tuple[list[dict[str, Any]], str]:
    path = split_path(config)
    identity = {
        "seed": config.split.seed,
        "fractions": list(fractions(config)),
        "labels_sha256": hashlib.sha256(canonical(labels)).hexdigest(),
        "images_md5": config.dataset.images_md5,
        "labels_md5": config.dataset.labels_md5,
        "algorithm": "PCG64 per-class shuffle, largest remainder v1",
    }
    if create:
        rows = partition_rows(labels, config.split.seed, fractions(config))
        content = split_bytes(rows)
        write_immutable(path, content)
        write_immutable(
            path.with_suffix(".json"),
            canonical({**identity, "sha256": hashlib.sha256(content).hexdigest()}),
        )
    if not path.is_file():
        raise FileNotFoundError("Frozen split missing; run data prepare once before experiments")
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    if reader.fieldnames != ["candidate_id", "index_no", "label", "split"]:
        raise ValueError("invalid frozen split columns")
    rows = [{**row, "index_no": int(row["index_no"]), "label": int(row["label"])} for row in reader]
    validate_rows(rows, labels, fractions(config))
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(content).hexdigest()
    if metadata != {**identity, "sha256": digest}:
        raise ValueError("frozen split checksum or identity mismatch")
    return rows, digest
