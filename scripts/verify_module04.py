"""Read-only checkpoint checks plus seven new validation-only pipeline smoke runs.

Run from the project root with the Edge_Project Python interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from edge_triage.config import AppConfig, load_config
from edge_triage.contracts import CandidateEvent, ModelTier, RunManifest
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.data.loaders import make_loader
from edge_triage.health import hash_file
from edge_triage.models.registry import ModelRegistry, read_index
from edge_triage.models.training import (
    evaluate,
    preliminary_latency,
    resolve_training_data,
    seed_everything,
)
from edge_triage.runtime.factory import run_smoke
from edge_triage.runtime.records import CandidateRecord


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/family_v1.yaml")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, hashes = resolve_training_data(load_config(args.config))
    torch.set_num_threads(config.training.cpu_threads)
    seed_everything(config.training.seed)
    index = read_index(Path(args.index))
    assert config.split.manifest is not None and config.preprocessing.artifact is not None
    registry = ModelRegistry(
        split_sha256=hash_file(config.split.manifest),
        preprocessing_sha256=hash_file(config.preprocessing.artifact),
        dataset_hashes=hashes,
        device=config.training.device,
    )
    validation = make_loader(config, "validation", run_id="module04-verification")
    tiers = {}
    for tier in ModelTier:
        entry = index["checkpoints"][tier.value]
        directory = Path(entry["directory"])
        registry.register(tier, entry["checkpoint_id"], directory, entry["manifest_sha256"])
        model, _ = registry.load(tier, entry["checkpoint_id"])
        metrics = json.loads((directory / "metrics.json").read_text())
        measured = evaluate(model, validation, registry.device)
        assert measured == metrics["validation"], (tier, measured, metrics["validation"])
        assert metrics["overfit_diagnostic"]["passed"]
        assert metrics["finite_logits_and_gradients"] and not metrics["test_evaluated"]
        tiers[tier.value] = {
            **entry,
            "model_state_sha256": hash_file(directory / "model_state.pt"),
            "metrics": metrics,
            "saved_checkpoint_validation_exact_match": True,
            "quiet_preliminary_latency": preliminary_latency(model, validation.dataset[0][1]),
        }
        print(f"Verified {tier.value}: AP={measured['average_precision']:.7f}", flush=True)
        del model
    runs = []
    for path in sorted((config.paths.project_root / "configs/experiments").glob("smoke*.yaml")):
        raw = load_config(path).model_dump()
        raw["models"]["checkpoint_index"] = Path(args.index).resolve()
        raw["pipeline"]["model_adapter"] = "cnn"
        directory = run_smoke(AppConfig.model_validate(raw), 20)
        manifest = RunManifest.model_validate_json((directory / "manifest.json").read_text())
        assert manifest.status == "completed"
        records = [
            CandidateRecord.model_validate(r)
            for r in pq.read_table(directory / "candidates.parquet").to_pylist()
        ]
        assert len(records) == 20 and all(r.status == "completed" for r in records)
        events = [
            CandidateEvent.model_validate_json(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        assert sum(e.event_type == "model.checkpoint.loaded" for e in events) == len(
            raw["models"]["tier_portfolio"]
        )
        runs.append(
            {
                "configuration": str(path),
                "directory": str(directory),
                "completed": 20,
                "failed": 0,
                "events": len(events),
                "manifest_sha256": hash_file(directory / "manifest.json"),
                "parquet_sha256": hash_file(directory / "candidates.parquet"),
            }
        )
        print(f"Verified {path.name}: 20 completed", flush=True)
    assert len(runs) == 7
    _, final_hashes = resolve_training_data(config)
    assert hashes == final_hashes
    write_immutable(
        Path(args.output),
        canonical(
            {
                "schema_version": 1,
                "dataset_hashes": hashes,
                "checkpoint_index_sha256": hash_file(Path(args.index)),
                "tiers": tiers,
                "runs": runs,
                "test_evaluated": False,
                "gate_b": (
                    "UNRESOLVED: small AP differences and overlapping batch1 latency; Module 05"
                ),
            }
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()
