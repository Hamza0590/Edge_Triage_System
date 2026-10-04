"""Binary Guo-style temperature scaling and hash-bound inference integration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from edge_triage.calibration.metrics import arrays, probabilities
from edge_triage.contracts import Contract, ModelTier, PredictionResult, p_real_from_logit
from edge_triage.health import hash_file
from edge_triage.models.registry import CheckpointManifest, RealModelAdapter
from edge_triage.runtime.interfaces import CandidateInput


def fit_temperature(
    labels: Any, logits: Any, *, partition: str, minimum: float = 0.01, maximum: float = 100
) -> dict[str, Any]:
    if partition != "validation":
        raise ValueError("temperature fitting is validation-only")
    y, z = arrays(labels, logits)
    if not 0 < minimum < 1 < maximum or not math.isfinite(maximum):
        raise ValueError("finite positive bounds must contain T=1")
    if len(np.unique(y)) != 2:
        raise ValueError("calibration requires both classes")
    # NLL is convex in inverse temperature beta. Bounded derivative bisection.
    lo, hi = 1 / maximum, 1 / minimum

    def gradient(beta: float) -> float:
        return float(np.mean((probabilities(z * beta) - y) * z))

    boundary = None
    if gradient(lo) >= 0:
        beta, boundary = lo, "maximum_temperature"
    elif gradient(hi) <= 0:
        beta, boundary = hi, "minimum_temperature"
    else:
        for _ in range(100):
            mid = (lo + hi) / 2
            if gradient(mid) > 0:
                hi = mid
            else:
                lo = mid
        beta = (lo + hi) / 2
    return {
        "temperature": 1 / beta,
        "objective": "mean binary NLL",
        "optimizer": "100-step float64 derivative bisection in inverse temperature",
        "bounds": [minimum, maximum],
        "boundary": boundary,
        "gradient_beta": gradient(beta),
    }


class CalibrationArtifact(Contract):
    schema_version: Literal[1] = 1
    artifact_id: str
    checkpoint_id: str
    checkpoint_manifest_sha256: str
    tier: ModelTier
    seed: int
    split_sha256: str
    preprocessing_sha256: str
    validation_rows_sha256: str
    prediction_sha256: str
    config_sha256: str
    code_sha256: str
    fit_partition: Literal["validation"] = "validation"
    temperature: float = Field(gt=0, allow_inf_nan=False)
    recommended_mode: Literal["uncalibrated", "calibrated"]
    selection_criterion: str
    fitting: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]


def load_calibration(
    path: Path, expected_sha256: str, manifest: CheckpointManifest, manifest_sha256: str
) -> CalibrationArtifact:
    if hash_file(path) != expected_sha256:
        raise ValueError("calibration artifact hash mismatch")
    result = CalibrationArtifact.model_validate_json(path.read_bytes())
    if (
        result.checkpoint_id != manifest.checkpoint_id
        or result.tier != manifest.tier
        or result.checkpoint_manifest_sha256 != manifest_sha256
        or result.split_sha256 != manifest.split_sha256
        or result.preprocessing_sha256 != manifest.preprocessing_sha256
    ):
        raise ValueError("incompatible checkpoint/calibration")
    return result


class CalibratedModelAdapter:
    def __init__(
        self, base: RealModelAdapter, artifact: CalibrationArtifact, mode: str = "recommended"
    ) -> None:
        if artifact.checkpoint_id != base.checkpoint_id or artifact.tier != base.model.spec.tier:
            raise ValueError("incompatible calibration adapter")
        if mode not in ("recommended", "uncalibrated", "calibrated"):
            raise ValueError("unknown probability mode")
        self.base, self.artifact = base, artifact
        self.mode = artifact.recommended_mode if mode == "recommended" else mode

    def predict(self, item: CandidateInput, tier: ModelTier) -> PredictionResult:
        result = self.base.predict(item, tier)
        if self.mode == "uncalibrated":
            return result
        return PredictionResult.model_validate(
            {
                **result.model_dump(),
                "calibrated_p_real": p_real_from_logit(result.logit / self.artifact.temperature),
                "calibration_artifact_id": self.artifact.artifact_id,
            }
        )
