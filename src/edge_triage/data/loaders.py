"""Read-only memory maps, frozen preprocessing and reproducible PyTorch loaders."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from edge_triage.config import AppConfig
from edge_triage.contracts import Candidate
from edge_triage.data.artifacts import PARTITIONS, frozen_split, load_raw
from edge_triage.data.preprocessing import crop_chw, load_profile, profile_identity


class MeerlichtDataset(Dataset):  # type: ignore[misc]
    def __init__(self, config: AppConfig, partition: str, *, run_id: str) -> None:
        if partition not in PARTITIONS:
            raise ValueError("unknown partition")
        if config.preprocessing.normalization != "train_standardize":
            raise ValueError("frozen training standardization is required")
        images, labels, hashes = load_raw(config)
        rows, split_hash = frozen_split(config, labels)
        self.profile = load_profile(config, profile_identity(config, split_hash, hashes))
        self.config, self.partition, self.run_id = config, partition, run_id
        self.rows = [row for row in rows if row["split"] == partition]
        self._images = images
        self.epoch = 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_images"] = None  # Windows spawn reopens mapping, never pickles raw images.
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __getitem__(self, position: int) -> tuple[Any, ...]:
        if self._images is None:
            self._images = np.load(self.config.dataset.images, mmap_mode="r", allow_pickle=False)
        row = self.rows[position]
        raw = self._images[row["index_no"]]
        pixels = crop_chw(raw, self.profile["crop_size"], tuple(self.profile["channels"]))
        if self.partition == "train":
            rng = np.random.default_rng([self.config.training.seed, self.epoch, row["index_no"]])
            if self.profile["horizontal_flip"] and rng.random() < 0.5:
                pixels = pixels[:, :, ::-1]
            if self.profile["vertical_flip"] and rng.random() < 0.5:
                pixels = pixels[:, ::-1, :]
        pixels = (
            pixels - np.array(self.profile["mean"], dtype=np.float32)[:, None, None]
        ) / np.array(self.profile["std"], dtype=np.float32)[:, None, None]
        if not np.isfinite(pixels).all():
            raise ValueError("non-finite transformed image")
        metadata = Candidate(
            run_id=self.run_id,
            trace_id=f"{self.run_id}:{row['candidate_id']}",
            candidate_id=row["candidate_id"],
            dataset_index=row["index_no"],
            arrival_monotonic_ns=0,
        )  # Offline reference; runtime assigns arrival time.
        result = (metadata, torch.from_numpy(np.ascontiguousarray(pixels)), int(row["label"]))
        return (*result, int(raw.nbytes)) if self.config.loader.include_original_bytes else result


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def collate_candidates(samples: list[tuple[Any, ...]]) -> dict[str, Any]:
    result = {
        "candidates": tuple(sample[0] for sample in samples),
        "images": torch.stack([sample[1] for sample in samples]),
        "labels": torch.tensor([sample[2] for sample in samples], dtype=torch.int64),
    }
    if len(samples[0]) == 4:
        result["original_bytes"] = torch.tensor(
            [sample[3] for sample in samples], dtype=torch.int64
        )
    return result


def make_loader(config: AppConfig, partition: str, *, run_id: str, epoch: int = 0) -> Any:
    dataset = MeerlichtDataset(config, partition, run_id=run_id)
    dataset.set_epoch(epoch)
    kwargs: dict[str, Any] = {}
    if config.loader.workers:
        kwargs.update(
            prefetch_factor=config.loader.prefetch_factor, multiprocessing_context="spawn"
        )
    return DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        shuffle=partition == "train",
        num_workers=config.loader.workers,
        pin_memory=config.loader.pin_memory,
        persistent_workers=False,
        collate_fn=collate_candidates,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.training.seed + epoch),
        **kwargs,
    )
