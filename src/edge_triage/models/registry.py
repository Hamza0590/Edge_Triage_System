"""Hash-verified, state-dictionary-only checkpoints and inference adapter."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field

from edge_triage.contracts import Contract, ModelTier, PredictionResult, p_real_from_logit
from edge_triage.event_logging import EventLogger
from edge_triage.health import hash_file
from edge_triage.models.cnn import ScalableRealBogusCNN, TierSpec, load_spec, resolve_device
from edge_triage.models.inference_state import INFERENCE_STATE
from edge_triage.runtime.interfaces import CandidateInput

CHECKPOINT_FILES = {
    "model_state.pt",
    "model_config.yaml",
    "training_config.yaml",
    "preprocessing_reference.json",
    "split_reference.json",
    "metrics.json",
    "history.csv",
}


class CheckpointManifest(Contract):
    schema_version: Literal[1] = 1
    checkpoint_id: str
    tier: ModelTier
    architecture_version: Literal["scalable_cnn_v1"]
    parameter_count: int = Field(gt=0)
    dataset_hashes: dict[str, str]
    split_sha256: str
    preprocessing_sha256: str
    seed: int = Field(ge=0)
    git_commit: str | None
    best_epoch: int = Field(ge=1)
    selection_metric: Literal["validation_average_precision"]
    file_hashes: dict[str, str]


class ModelRegistry:
    def __init__(
        self,
        *,
        split_sha256: str,
        preprocessing_sha256: str,
        dataset_hashes: dict[str, str],
        device: str = "cpu",
        logger: EventLogger | None = None,
    ) -> None:
        self.split_sha256 = split_sha256
        self.preprocessing_sha256 = preprocessing_sha256
        self.dataset_hashes = dataset_hashes
        self.device = resolve_device(device)
        self.logger = logger
        self.entries: dict[tuple[ModelTier, str], tuple[Path, str]] = {}

    def register(
        self, tier: ModelTier, checkpoint_id: str, directory: Path, manifest_sha256: str
    ) -> None:
        key = (tier, checkpoint_id)
        if key in self.entries:
            raise ValueError("checkpoint already registered")
        self.entries[key] = (directory, manifest_sha256)

    def load(
        self, tier: ModelTier, checkpoint_id: str, *, allow_incompatible_diagnostic: bool = False
    ) -> tuple[ScalableRealBogusCNN, TierSpec]:
        directory, digest = self.entries[(tier, checkpoint_id)]
        path = directory / "checkpoint_manifest.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("checkpoint manifest hash mismatch")
        manifest = CheckpointManifest.model_validate_json(raw)
        if manifest.tier != tier or manifest.checkpoint_id != checkpoint_id:
            raise ValueError("checkpoint identity mismatch")
        if set(manifest.file_hashes) != CHECKPOINT_FILES:
            raise ValueError("checkpoint file inventory mismatch")
        for name, expected in manifest.file_hashes.items():
            if hash_file(directory / name) != expected:
                raise ValueError(f"checkpoint file hash mismatch: {name}")
        compatible = (
            manifest.split_sha256 == self.split_sha256
            and manifest.preprocessing_sha256 == self.preprocessing_sha256
            and manifest.dataset_hashes == self.dataset_hashes
        )
        if not compatible and not allow_incompatible_diagnostic:
            raise ValueError("incompatible dataset/split/preprocessing")
        spec = load_spec(directory / "model_config.yaml")
        if spec.tier != tier or spec.architecture_version != manifest.architecture_version:
            raise ValueError("incompatible architecture")
        model = ScalableRealBogusCNN(spec)
        if model.parameter_count != manifest.parameter_count:
            raise ValueError("checkpoint parameter count mismatch")
        model.load_state_dict(
            torch.load(directory / "model_state.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("non-finite checkpoint tensor")
        model.to(self.device).eval()
        if self.logger:
            self.logger.emit(
                module="models",
                event_type="model.checkpoint.loaded",
                reason_code="HASH_VERIFIED",
                message="Checkpoint loaded",
                payload={
                    "checkpoint_id": checkpoint_id,
                    "tier": tier.value,
                    "compatible": compatible,
                    "device": str(self.device),
                },
            )
        return model, spec


class RealModelAdapter:
    def __init__(self, model: ScalableRealBogusCNN, checkpoint_id: str) -> None:
        self.model, self.checkpoint_id = model, checkpoint_id
        self.device = next(model.parameters()).device
        with INFERENCE_STATE.write():
            self.model.eval()

    def predict(self, item: CandidateInput, tier: ModelTier) -> PredictionResult:
        if tier != self.model.spec.tier:
            raise ValueError("adapter tier mismatch")
        started = time.monotonic_ns()
        with INFERENCE_STATE.read(), torch.inference_mode():
            if any(module.training for module in self.model.modules()):
                raise RuntimeError("serving model must remain in evaluation mode")
            logit = float(self.model(item.image.unsqueeze(0).to(self.device)).item())
        return PredictionResult(
            candidate_id=item.candidate.candidate_id,
            tier=tier,
            checkpoint_id=self.checkpoint_id,
            logit=logit,
            p_real=p_real_from_logit(logit),
            inference_ms=(time.monotonic_ns() - started) / 1e6,
        )


def read_index(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid checkpoint index")
    return value
