import itertools
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from edge_triage.config import AppConfig, load_config


@pytest.mark.parametrize(
    "raw",
    [
        {"typo": True},
        {"training": {"epohs": 10}},
        {"models": {"tier_portfolio": []}},
        {"models": {"tier_portfolio": ["tiny", "tiny"]}},
        {"models": {"tier_portfolio": ["large", "tiny"]}},
        {"split": {"train_fraction": 0.8}},
        {"split": {"seed": -1}},
        {"calibration": {"fit_partition": "test"}},
        {"threshold": {"fit_partition": "test"}},
        {"admission": {"fixed_concurrency": 5, "max_concurrency": 2}},
        {"models": {"tier_portfolio": ["large"]}, "tier_selection": {"policy": "fixed"}},
        {"dataset": {"images_md5": "a" * 32}},
    ],
)
def test_invalid_config(raw):
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_all_portfolios_and_execution_modes():
    for length in (1, 2, 3):
        for tiers in itertools.combinations(("tiny", "medium", "large"), length):
            for mode in ("sequential", "fixed_concurrency", "dynamic_concurrency"):
                config = AppConfig.model_validate(
                    {"models": {"tier_portfolio": tiers}, "admission": {"execution_mode": mode}}
                )
                assert len(config.models.tier_portfolio) == length


def test_path_resolution_independent_of_cwd(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "settings.yaml"
    path.write_text("paths:\n  project_root: ..\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert config.paths.project_root == tmp_path
    assert config.dataset.images == tmp_path / "MeerLICHT_images.npy"


def test_stable_hash_round_trip_and_sensitive_to_seed(config, tmp_path):
    path = tmp_path / "resolved.yaml"
    path.write_text(config.resolved_yaml(), encoding="utf-8")
    assert load_config(path).config_hash == config.config_hash
    raw = config.model_dump(mode="json")
    reversed_mapping = dict(reversed(list(raw.items())))
    assert AppConfig.model_validate(reversed_mapping).config_hash == config.config_hash
    raw["experiment"]["seed"] += 1
    assert AppConfig.model_validate(raw).config_hash != config.config_hash


@pytest.mark.parametrize(
    "text", ["[]", "", "training:\n  epochs: 1\n  epochs: 2\n", "paths:\n  runs_dir: .\n"]
)
def test_invalid_yaml_and_unsafe_output_paths(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_resolved_config_is_serializable(config):
    assert yaml.safe_load(config.resolved_yaml())["paths"]["project_root"] == str(
        Path(__file__).resolve().parents[2]
    )
