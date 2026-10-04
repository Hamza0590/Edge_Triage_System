import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from edge_triage.config import AppConfig
from edge_triage.data.artifacts import partition_rows, split_bytes, validate_rows, write_immutable
from edge_triage.data.preprocessing import crop_chw, fit_statistics


def test_crop_and_explicit_channel_order():
    values = np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    actual = crop_chw(values, 4, ("difference", "new"))
    np.testing.assert_array_equal(actual[0], values[2:6, 2:6, 2])
    np.testing.assert_array_equal(actual[1], values[2:6, 2:6, 0])
    assert actual.flags.c_contiguous and actual.dtype == np.float32


@pytest.mark.parametrize("channels", [("new", "new"), ("unknown",), ()])
def test_bad_channel_selection(channels):
    with pytest.raises(ValueError):
        crop_chw(np.zeros((8, 8, 3)), 4, channels)


def test_wrong_source_channel_configuration():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"dataset": {"channel_order": ["difference", "new", "reference"]}})


def test_train_only_statistics_ignore_poisoned_holdouts():
    images = np.arange(4 * 8 * 8 * 3, dtype=np.float32).reshape(4, 8, 8, 3)
    rows = [
        {"index_no": i, "split": split}
        for i, split in enumerate(("train", "train", "validation", "test"))
    ]
    baseline = fit_statistics(images, rows, 4, ("new", "reference", "difference"))
    images[2:] = np.nan
    assert fit_statistics(images, rows, 4, ("new", "reference", "difference")) == baseline
    expected = images[:2, 2:6, 2:6].astype(np.float64)
    np.testing.assert_allclose(baseline["mean"], expected.mean(axis=(0, 1, 2)))
    np.testing.assert_allclose(baseline["std"], expected.std(axis=(0, 1, 2)))
    assert baseline["fit_rows"] == 2


def test_split_determinism_coverage_stratification_and_seed_versions(tmp_path):
    labels = [0] * 1572 + [1] * 1647
    fractions = (0.5, 0.2, 0.3)
    rows = partition_rows(labels, 20260924, fractions)
    validate_rows(rows, labels, fractions)
    content = split_bytes(rows)
    assert content == split_bytes(partition_rows(labels, 20260924, fractions))
    assert (
        hashlib.sha256(content).digest()
        != hashlib.sha256(split_bytes(partition_rows(labels, 42, fractions))).digest()
    )
    assert len({row["index_no"] for row in rows}) == 3219
    path = tmp_path / "split.csv"
    write_immutable(path, content)
    before = path.stat().st_mtime_ns
    write_immutable(path, content)
    assert path.stat().st_mtime_ns == before
    with pytest.raises(FileExistsError):
        write_immutable(path, b"changed")
    assert path.read_bytes() == content
    with pytest.raises(ValueError):
        validate_rows([rows[1], *rows[1:]], labels, fractions)
