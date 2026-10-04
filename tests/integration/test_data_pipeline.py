import json
from pathlib import Path

import numpy as np
import pytest
import torch

from edge_triage.config import load_config
from edge_triage.data.artifacts import frozen_split, load_raw
from edge_triage.data.audit import audit
from edge_triage.data.loaders import MeerlichtDataset, make_loader
from edge_triage.data.preprocessing import crop_chw, prepare
from edge_triage.health import dataset_hashes

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    config = load_config(ROOT / "configs/data/meerlicht.yaml")
    raw = config.model_dump()
    raw["paths"]["artifacts_dir"] = tmp_path_factory.mktemp("prepared")
    config = type(config).model_validate(raw)
    paths = (config.dataset.images, config.dataset.labels)
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    summary = prepare(config)
    yield config, summary
    assert len(dataset_hashes(config)) == 2
    assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


@pytest.mark.dataset
def test_full_audit_and_repeatable_artifacts(prepared):
    config, summary = prepared
    report = audit(config)
    assert report["npy"]["shape"] == [3219, 100, 100, 3]
    assert report["npy"]["dtype"] == "float32"
    assert report["class_counts"] == {"Real": 1647, "Bogus": 1572}
    assert set(report["quality"].values()) == {0}
    assert Path(summary["visualization_path"]).is_file()
    assert prepare(config) == summary
    profile = json.loads(Path(summary["preprocessing_path"]).read_text())
    assert profile["fit_partition"] == "train" and profile["fit_rows"] == 1610
    assert sum(sum(counts.values()) for counts in summary["counts"].values()) == 3219


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_dataset_shape_labels_bytes_and_no_runtime_label(prepared, partition):
    config, _ = prepared
    dataset = MeerlichtDataset(config, partition, run_id="test")
    metadata, tensor, label, size = dataset[0]
    assert metadata.label is None
    assert tensor.shape == (3, 30, 30) and tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all() and label in (0, 1) and size == 120000
    if partition != "train":
        dataset.set_epoch(20)
        assert torch.equal(dataset[0][1], tensor)
        raw_image = np.load(config.dataset.images, mmap_mode="r")[metadata.dataset_index]
        expected = crop_chw(raw_image, 30, ("new", "reference", "difference"))
        expected = (
            expected - np.array(dataset.profile["mean"], dtype=np.float32)[:, None, None]
        ) / np.array(dataset.profile["std"], dtype=np.float32)[:, None, None]
        np.testing.assert_array_equal(tensor.numpy(), expected)


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_repeated_loaders_all_batches_identical(prepared, partition):
    config, _ = prepared
    first = list(make_loader(config, partition, run_id="repeat"))
    second = list(make_loader(config, partition, run_id="repeat"))
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a["images"], b["images"])
        assert torch.equal(a["labels"], b["labels"])
        assert a["candidates"] == b["candidates"]


def test_spawn_workers_match_single_worker_training(prepared):
    config, _ = prepared
    raw = config.model_dump()
    raw["loader"]["workers"] = 2
    workers = type(config).model_validate(raw)
    single = list(make_loader(config, "train", run_id="worker", epoch=2))
    multiple = list(make_loader(workers, "train", run_id="worker", epoch=2))
    for a, b in zip(single, multiple, strict=True):
        assert torch.equal(a["images"], b["images"])
        assert torch.equal(a["labels"], b["labels"])


def test_loader_refuses_modified_split_and_missing_profile(prepared, tmp_path):
    config, _ = prepared
    images, labels, _ = load_raw(config)
    assert images.flags.writeable is False
    invalid = tmp_path / "changed.csv"
    invalid.write_text("changed", encoding="utf-8")
    raw = config.model_dump()
    raw["split"]["manifest"] = invalid
    with pytest.raises(ValueError, match="frozen split"):
        frozen_split(type(config).model_validate(raw), labels)
    raw = config.model_dump()
    raw["preprocessing"]["artifact"] = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        MeerlichtDataset(type(config).model_validate(raw), "test", run_id="missing")
