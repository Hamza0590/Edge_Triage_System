from pathlib import Path

import pytest

from edge_triage.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(PROJECT_ROOT / "configs/default.yaml")


@pytest.fixture
def isolated_config(config, tmp_path):
    raw = config.model_dump(mode="python")
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    return type(config).model_validate(raw)
