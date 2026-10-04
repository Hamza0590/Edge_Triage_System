"""Sequential dropout sampling with exact state and diagnostic RNG restoration."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from torch import nn

from edge_triage.contracts import PredictionResult, UncertaintyResult
from edge_triage.models.inference_state import INFERENCE_STATE

_SAMPLING_LOCK = threading.RLock()
_DROPOUT = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


class SamplingError(RuntimeError):
    """Failed sampling retains attempted/completed forwards for cost accounting."""

    def __init__(self, attempted: int, completed: int, duration_ms: float) -> None:
        super().__init__("MC sampling failed; see chained exception")
        self.attempted = attempted
        self.completed = completed
        self.duration_ms = duration_ms


@contextmanager
def dropout_inference(model: Any) -> Iterator[None]:
    """Keep BatchNorm frozen; restore even mixed original module states on failure.

    Package deterministic adapters take the corresponding read guard. Exclusive
    sampling also protects diagnostic RNG save/restore across different models.
    """
    with _SAMPLING_LOCK, INFERENCE_STATE.write():
        states = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            for module, _ in states:
                if isinstance(module, _DROPOUT):
                    module.train(True)
            with torch.inference_mode():
                yield
        finally:
            for module, training in states:
                module.training = training


def sample_probabilities(
    model: Any,
    image: Any,
    passes: int,
    *,
    temperature: float = 1,
    seed: int | None = None,
) -> tuple[list[float], float]:
    if passes < 2 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("MC sampling requires >=2 passes and finite positive temperature")
    if seed is not None and not 0 <= seed < 2**63:
        raise ValueError("diagnostic seed must be in [0, 2**63)")
    if not any(isinstance(m, _DROPOUT) and m.p > 0 for m in model.modules()):
        raise ValueError("MC sampling requires active dropout layers")
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    start = time.perf_counter_ns()
    values: list[float] = []
    attempted = 0
    try:
        with _SAMPLING_LOCK, dropout_inference(model):
            with torch.random.fork_rng(devices=devices) if seed is not None else nullcontext():
                if seed is not None:
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(seed)
                batch = image.unsqueeze(0).to(device)
                for _ in range(passes):
                    attempted += 1
                    value = float(torch.sigmoid(model(batch).double() / temperature).item())
                    if not math.isfinite(value):
                        raise ValueError("nonfinite MC probability")
                    values.append(value)
    except Exception as error:
        raise SamplingError(
            attempted, len(values), (time.perf_counter_ns() - start) / 1e6
        ) from error
    # item() synchronizes each CUDA result; includes transfer, RNG and state costs.
    return values, (time.perf_counter_ns() - start) / 1e6


def summarize(
    prediction: PredictionResult,
    values: Sequence[float],
    duration_ms: float,
) -> UncertaintyResult:
    if len(values) < 2 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("at least two finite probabilities required")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((p - mean) ** 2 for p in values) / len(values)
    entropy = -sum(p * math.log(p) for p in (mean, 1 - mean) if p > 0)
    votes = sum(p >= 0.5 for p in values)
    return UncertaintyResult(
        candidate_id=prediction.candidate_id,
        tier=prediction.tier,
        method="mc_dropout",
        passes=len(values),
        additional_forward_passes=len(values),
        mean_p_real=mean,
        variance=variance,
        standard_deviation=math.sqrt(variance),
        predictive_entropy=entropy,
        variation_ratio=min(votes, len(values) - votes) / len(values),
        duration_ms=duration_ms,
        reason_code="MC_SAMPLED",
    )
