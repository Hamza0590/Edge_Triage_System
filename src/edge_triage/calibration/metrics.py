"""Saved-logit evaluation; tie-aware ranking and explicit undefined metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def arrays(labels: Any, logits: Any) -> tuple[Any, Any]:
    y, z = np.asarray(labels, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if y.ndim != 1 or z.shape != y.shape or not len(y):
        raise ValueError("expected nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(z).all():
        raise ValueError("binary labels and finite logits required")
    return y, z


def probabilities(logits: Any, temperature: float = 1) -> Any:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    return np.exp(-np.logaddexp(0, -z))


def metrics(
    labels: Any, logits: Any, *, temperature: float = 1, threshold_logit: float = 0, bins: int = 15
) -> dict[str, Any]:
    y, z = arrays(labels, logits)
    if bins < 1 or not math.isfinite(threshold_logit):
        raise ValueError("positive bins and finite threshold required")
    p = probabilities(z, temperature)
    predicted = z >= threshold_logit  # Avoid rounded sigmoid ties at extreme logits.
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn, tn = int(y.sum()) - tp, int((1 - y).sum()) - fp
    undefined: dict[str, str] = {}

    def ratio(name: str, a: float, b: float) -> float | None:
        if b == 0:
            undefined[name] = "zero denominator"
            return None
        return a / b

    result: dict[str, Any] = {
        "n": len(y),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": ratio("precision", tp, tp + fp),
        "recall": ratio("recall", tp, tp + fn),
        "f1": ratio("f1", 2 * tp, 2 * tp + fp + fn),
        "fnr": ratio("fnr", fn, tp + fn),
        "fpr": ratio("fpr", fp, tn + fp),
        "mcc": ratio(
            "mcc", tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        ),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(np.mean(np.logaddexp(0, z / temperature) - y * z / temperature)),
        "threshold_raw_logit": threshold_logit,
        "threshold_probability": float(probabilities(threshold_logit, temperature)),
    }
    order = np.argsort(-z, kind="stable")
    sy, sz = y[order], z[order]
    ends = np.r_[np.flatnonzero(np.diff(sz)), len(y) - 1]
    tps = np.cumsum(sy)[ends]
    fps = ends + 1 - tps
    if y.sum() == 0:
        result["average_precision"] = result["pr_auc_trapezoidal"] = None
        undefined["average_precision"] = undefined["pr_auc_trapezoidal"] = "no positives"
    else:
        recall = tps / y.sum()
        precision = tps / (ends + 1)
        result["average_precision"] = float(np.sum(np.diff(np.r_[0, recall]) * precision))
        result["pr_auc_trapezoidal"] = float(np.trapezoid(np.r_[1, precision], np.r_[0, recall]))
    if len(np.unique(y)) < 2:
        result["roc_auc"] = None
        undefined["roc_auc"] = "requires both classes"
    else:
        result["roc_auc"] = float(
            np.trapezoid(np.r_[0, tps / y.sum()], np.r_[0, fps / (1 - y).sum()])
        )
    # Binary classwise ECE, not top-label confidence ECE.
    assignment = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        mask = assignment == index
        count = int(mask.sum())
        mean_p = float(p[mask].mean()) if count else None
        frequency = float(y[mask].mean()) if count else None
        if count:
            assert mean_p is not None and frequency is not None
            ece += count / len(y) * abs(float(mean_p) - float(frequency))
        reliability.append(
            {"bin": index, "count": count, "mean_p_real": mean_p, "real_frequency": frequency}
        )
    result.update(ece=ece, reliability=reliability, undefined=undefined)
    return result


def recall_threshold(labels: Any, logits: Any, target: float) -> float:
    y, z = arrays(labels, logits)
    if not 0 < target <= 1 or not y.sum():
        raise ValueError("positive recall target and positive examples required")
    # Highest inclusive threshold attaining target recall; all ties admitted.
    return float(np.sort(z[y == 1])[::-1][math.ceil(target * int(y.sum())) - 1])


def bootstrap(
    labels: Any,
    logits: Any,
    *,
    temperature: float,
    threshold: float,
    bins: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    y, z = arrays(labels, logits)
    rng = np.random.default_rng(seed)
    names = (
        "precision",
        "recall",
        "f1",
        "mcc",
        "fnr",
        "fpr",
        "roc_auc",
        "average_precision",
        "pr_auc_trapezoidal",
        "brier",
        "nll",
        "ece",
    )
    values: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        row = metrics(
            y[indices], z[indices], temperature=temperature, threshold_logit=threshold, bins=bins
        )
        for name in names:
            if row[name] is not None:
                values[name].append(row[name])
    return {
        "method": "IID candidate percentile 95%; fixed fitted T/threshold; no refit",
        "limitation": "fit-set conditional intervals; no selection/group/seed uncertainty",
        "seed": seed,
        "repeats": repeats,
        "intervals": {
            name: {
                "low": float(np.percentile(v, 2.5)) if v else None,
                "high": float(np.percentile(v, 97.5)) if v else None,
                "valid_replicates": len(v),
            }
            for name, v in values.items()
        },
    }
