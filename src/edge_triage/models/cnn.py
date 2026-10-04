"""Versioned same-family CNNs. Forward returns logits, never probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import Field
from torch import nn

from edge_triage.config import UniqueKeyLoader
from edge_triage.contracts import Contract, ModelTier, PositiveCount


class TierSpec(Contract):
    architecture_version: Literal["scalable_cnn_v1"] = "scalable_cnn_v1"
    tier: ModelTier
    base_filters: PositiveCount
    dense_width: PositiveCount
    input_channels: PositiveCount = 3
    crop_size: int = Field(default=30, ge=8, le=100, strict=True)
    spatial_dropout: float = Field(default=0.2, ge=0, lt=1)
    dense_dropout: float = Field(default=0.5, ge=0, lt=1)


def load_spec(path: Path) -> TierSpec:
    return TierSpec.model_validate(yaml.load(path.read_text("utf-8"), Loader=UniqueKeyLoader))


class ScalableRealBogusCNN(nn.Module):  # type: ignore[misc]
    def __init__(self, spec: TierSpec) -> None:
        super().__init__()
        self.spec = spec
        layers: list[Any] = []
        channels = spec.input_channels
        for width in (spec.base_filters, 2 * spec.base_filters, 4 * spec.base_filters):
            layers.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(spec.spatial_dropout),
                ]
            )
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * (spec.crop_size // 8) ** 2, spec.dense_width),
            nn.ReLU(),
            nn.Dropout(spec.dense_dropout),
            nn.Linear(spec.dense_width, 1),
        )

    def forward(self, images: Any) -> Any:
        if images.ndim != 4 or tuple(images.shape[1:]) != (
            self.spec.input_channels,
            self.spec.crop_size,
            self.spec.crop_size,
        ):
            raise ValueError("model input must match NCHW architecture specification")
        return self.classifier(self.features(images)).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def operation_counts(self) -> dict[str, int | str]:
        """Exact dense Conv/Linear MAC count, excluding BN, activation, pooling and bias."""
        size, channels, macs = self.spec.crop_size, self.spec.input_channels, 0
        for width in (
            self.spec.base_filters,
            2 * self.spec.base_filters,
            4 * self.spec.base_filters,
        ):
            macs += size * size * width * channels * 9
            size //= 2
            channels = width
        macs += channels * size * size * self.spec.dense_width + self.spec.dense_width
        return {
            "conv_linear_macs": macs,
            "conv_linear_flops": 2 * macs,
            "scope": "batch1; Conv/Linear only; one multiply-add = 2 FLOPs",
        }


def resolve_device(requested: str) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )
