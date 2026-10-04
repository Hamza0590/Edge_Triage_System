import json
import shutil
from itertools import combinations
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import CandidateEvent, ModelTier
from edge_triage.health import hash_file
from edge_triage.models import training
from edge_triage.models.cnn import ScalableRealBogusCNN
from edge_triage.models.registry import ModelRegistry
from edge_triage.runtime.factory import run_smoke

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("checkpoints")
    raw = load_config(ROOT / "configs/training/family_v1.yaml").model_dump()
    raw["paths"]["artifacts_dir"] = root / "artifacts"
    raw["paths"]["runs_dir"] = root / "runs"
    raw["preprocessing"]["artifact"] = (
        ROOT / "artifacts/data/preprocessing_v1_c3436bda1c7877ab.json"
    )
    raw["training"].update(epochs=2, batch_size=8, device="cpu", debug_overfit=False)
    raw["loader"]["pin_memory"] = False
    config = AppConfig.model_validate(raw)
    original = training.make_loader
    accessed = []

    def small_loader(config, partition, **kwargs):
        assert partition in ("train", "validation")
        accessed.append(partition)
        loader = original(config, partition, **kwargs)
        rows = loader.dataset.rows
        loader.dataset.rows = [
            r for label in (0, 1) for r in [r for r in rows if r["label"] == label][:4]
        ]
        return loader

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training, "make_loader", small_loader)
        index_path = training.train_family(config)
    assert accessed == ["train", "validation"] * 3
    return config, json.loads(index_path.read_text()), index_path


def registry_for(config, entry, tier):
    resolved, hashes = training.resolve_training_data(config)
    registry = ModelRegistry(
        split_sha256=hash_file(resolved.split.manifest),
        preprocessing_sha256=hash_file(resolved.preprocessing.artifact),
        dataset_hashes=hashes,
    )
    registry.register(
        tier, entry["checkpoint_id"], Path(entry["directory"]), entry["manifest_sha256"]
    )
    return registry


@pytest.mark.parametrize("tier", list(ModelTier))
def test_checkpoints_equivalent_complete_and_logged(checkpoints, tier):
    config, index, _ = checkpoints
    entry = index["checkpoints"][tier.value]
    directory = Path(entry["directory"])
    registry = registry_for(config, entry, tier)
    model, spec = registry.load(tier, entry["checkpoint_id"])
    direct = ScalableRealBogusCNN(spec).eval()
    direct.load_state_dict(torch.load(directory / "model_state.pt", weights_only=True))
    x = torch.randn(3, 3, 30, 30)
    with torch.inference_mode():
        assert torch.equal(model(x), direct(x))
    assert not model.training
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["test_evaluated"] is False
    assert metrics["train_examples"] == metrics["validation_examples"] == 8
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
    for name, digest in manifest["file_hashes"].items():
        assert hash_file(directory / name) == digest
    events = [
        CandidateEvent.model_validate_json(line)
        for line in (config.paths.runs_dir / entry["run_id"] / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    types = {e.event_type for e in events}
    assert {
        "model.training.started",
        "model.epoch.started",
        "model.epoch.completed",
        "model.checkpoint.saved",
        "model.training.completed",
        "run.completed",
    } <= types
    run_manifest = json.loads(
        (config.paths.runs_dir / entry["run_id"] / "manifest.json").read_text()
    )
    assert run_manifest["checkpoint_hashes"][0]["digest"] == entry["manifest_sha256"]


def test_registry_tampering_and_compatibility(checkpoints, tmp_path):
    config, index, _ = checkpoints
    entry = dict(index["checkpoints"]["tiny"])
    registry = registry_for(config, entry, ModelTier.TINY)
    registry.split_sha256 = "0" * 64
    with pytest.raises(ValueError, match="incompatible"):
        registry.load(ModelTier.TINY, entry["checkpoint_id"])
    registry.load(ModelTier.TINY, entry["checkpoint_id"], allow_incompatible_diagnostic=True)
    directory = tmp_path / "tampered"
    shutil.copytree(entry["directory"], directory)
    entry["directory"] = str(directory)
    registry = registry_for(config, entry, ModelTier.TINY)
    (directory / "model_state.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file hash mismatch"):
        registry.load(ModelTier.TINY, entry["checkpoint_id"])
    (directory / "checkpoint_manifest.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        registry.load(ModelTier.TINY, entry["checkpoint_id"])


PORTFOLIOS = [p for size in (1, 2, 3) for p in combinations(list(ModelTier), size)]


@pytest.mark.parametrize("portfolio", PORTFOLIOS)
def test_real_pipeline_all_portfolios(checkpoints, tmp_path, portfolio):
    config, _, index_path = checkpoints
    raw = load_config(ROOT / "configs/experiments/smoke.yaml").model_dump()
    raw["paths"]["runs_dir"] = tmp_path / "runs"
    raw["pipeline"]["model_adapter"] = "cnn"
    raw["models"].update(tier_portfolio=portfolio, checkpoint_index=index_path)
    raw["tier_selection"]["fixed_tier"] = portfolio[0]
    raw["training"]["device"] = "cpu"
    run = run_smoke(AppConfig.model_validate(raw), limit=3)
    records = pq.read_table(run / "candidates.parquet").to_pylist()
    assert len(records) == 3 and all(r["status"] == "completed" for r in records)
    assert all(r["final_p_real"] >= 0 for r in records)
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["checkpoint_hashes"]) == len(portfolio)
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert sum(e["event_type"] == "model.checkpoint.loaded" for e in events) == len(portfolio)


def test_training_rejects_test_evaluation():
    class Loader:
        class dataset:
            partition = "test"

    with pytest.raises(ValueError, match="restricted to validation"):
        training.evaluate(None, Loader(), "cpu")
