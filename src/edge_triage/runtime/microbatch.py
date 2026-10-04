"""Fixed micro-batch baseline with an explicit oldest-arrival waiting deadline."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import torch

from edge_triage.models.inference_state import INFERENCE_STATE


def replay_batches(
    model: Any,
    images: Any,
    arrivals: Sequence[float],
    batch_size: int,
    max_wait_s: float,
) -> dict[str, Any]:
    """Replay preprocessed inputs. Times include queue formation but exclude durable logging.

    Arrivals are relative seconds in nondecreasing order. Every output contains its
    input position for identity joins; labels are never accepted by this function.
    """
    if (
        batch_size not in (1, 2, 4, 8)
        or not math.isfinite(max_wait_s)
        or max_wait_s < 0
        or len(images) != len(arrivals)
    ):
        raise ValueError("invalid micro-batch configuration")
    if (
        not arrivals
        or any(not math.isfinite(a) or a < 0 for a in arrivals)
        or list(arrivals) != sorted(arrivals)
    ):
        raise ValueError("ordered nonnegative arrivals required")
    device = next(model.parameters()).device
    origin = time.perf_counter()
    rows = []
    batches = []
    index = 0
    while index < len(arrivals):
        time.sleep(max(0, origin + arrivals[index] - time.perf_counter()))
        deadline = origin + arrivals[index] + max_wait_s
        stop = index + 1
        while stop < len(arrivals) and stop - index < batch_size:
            due = origin + arrivals[stop]
            if due > max(deadline, time.perf_counter()):
                time.sleep(max(0, deadline - time.perf_counter()))
                break
            time.sleep(max(0, due - time.perf_counter()))
            stop += 1
        start = time.perf_counter()
        with INFERENCE_STATE.read(), torch.inference_mode():
            logits = model(torch.as_tensor(images[index:stop].copy()).to(device)).cpu().tolist()
        end = time.perf_counter()
        batches.append({"size": stop - index, "forward_ms": (end - start) * 1000})
        for offset, logit in enumerate(logits, index):
            rows.append(
                {
                    "position": offset,
                    "logit": logit,
                    "wait_ms": (start - origin - arrivals[offset]) * 1000,
                    "end_to_end_ms": (end - origin - arrivals[offset]) * 1000,
                }
            )
        index = stop
    elapsed = time.perf_counter() - origin
    return {
        "rows": rows,
        "batches": batches,
        "elapsed_s": elapsed,
        "throughput_per_s": len(rows) / elapsed,
        "batch_size": batch_size,
        "max_wait_ms": max_wait_s * 1000,
        "scope": "preprocessed component replay; excludes durable pipeline instrumentation",
    }
