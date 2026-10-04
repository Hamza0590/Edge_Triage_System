from pathlib import Path

import numpy as np
import pytest
import torch

from edge_triage.config import AppConfig, load_config
from edge_triage.health import hash_file
from edge_triage.runtime.microbatch import replay_batches
from edge_triage.scheduling.contention import ContentionProfile, load_contention_cap

ROOT = Path(__file__).resolve().parents[2]


class MeanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, images):
        return images.mean(dim=(1, 2, 3))*self.weight


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_microbatch_ids_wait_and_predictions(batch):
    images = np.arange(11*3*30*30, dtype=np.float32).reshape(11, 3, 30, 30)
    model = MeanModel().eval()
    result = replay_batches(model, images, [0.0]*11, batch, .002)
    assert [r["position"] for r in result["rows"]] == list(range(11))
    assert [r["logit"] for r in result["rows"]] == pytest.approx(images.mean((1, 2, 3)))
    assert sum(b["size"] for b in result["batches"]) == 11
    assert all(b["size"] <= batch for b in result["batches"])
    assert all(r["end_to_end_ms"] >= r["wait_ms"] >= 0 for r in result["rows"])


def test_microbatch_deadline_and_invalid_arrivals():
    model = MeanModel().eval()
    images = np.zeros((3, 3, 30, 30), dtype=np.float32)
    result = replay_batches(model, images, [0, .03, .06], 8, .002)
    assert [b["size"] for b in result["batches"]] == [1, 1, 1]
    for arrivals in ([0, -1, 1], [0, 2, 1], [0, float("nan"), 1]):
        with pytest.raises(ValueError):
            replay_batches(model, images, arrivals, 8, .002)


def test_contention_identity_checksum_and_unknown_conservative_cap(tmp_path):
    config = load_config(ROOT/"configs/experiments/module09_v1.yaml")
    profile = ContentionProfile(
        hardware_id=config.scheduler.hardware_id, device="cpu",
        family_sha256=hash_file(config.models.checkpoint_index),
        profiles_sha256=config.scheduler.profiles_sha256,
        benchmark_sha256="a"*64, criterion="test-only", caps={"tiny+medium+large": 2},
    )
    path = tmp_path/"contention.json"
    path.write_text(profile.model_dump_json())
    raw = config.model_dump()
    raw["scheduler"].update(contention_profile=path, contention_sha256=hash_file(path))
    assert load_contention_cap(AppConfig.model_validate(raw)) == 2
    raw["models"]["tier_portfolio"] = ("tiny",)
    assert load_contention_cap(AppConfig.model_validate(raw)) == 1
    raw["scheduler"]["hardware_id"] = "wrong"
    with pytest.raises(ValueError, match="incompatible"):
        load_contention_cap(AppConfig.model_validate(raw))
    raw["scheduler"]["hardware_id"] = config.scheduler.hardware_id
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="checksum"):
        load_contention_cap(AppConfig.model_validate(raw))
