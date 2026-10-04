"""Validation-only fitting and saved-sample diagnostics; never used by runtime policies."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from edge_triage.calibration.metrics import metrics


def fit_interval(probabilities: Any, fraction: float, *, partition: str) -> tuple[float, float]:
    p = np.asarray(probabilities, dtype=float)
    if partition != "validation":
        raise ValueError("routing fitting is validation-only")
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("nonempty finite probability vector required")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    radius = float(np.sort(np.abs(p - 0.5))[math.ceil(fraction * len(p)) - 1])
    # Outward rounding preserves the requested inclusive rank under float subtraction.
    return max(0.0, float(np.nextafter(0.5 - radius, -np.inf))), min(
        1.0, float(np.nextafter(0.5 + radius, np.inf))
    )


def fit_std(values: Any, quantile: float, *, partition: str) -> float:
    values = np.asarray(values, dtype=float)
    if partition != "validation":
        raise ValueError("routing fitting is validation-only")
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not ((values >= 0) & (values <= 0.5)).all()
        or not 0 <= quantile <= 1
    ):
        raise ValueError("finite population standard deviations and valid quantile required")
    return float(np.quantile(values, quantile, method="linear"))


def probability_metrics(y: Any, p: Any) -> dict[str, Any]:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    result = metrics(y, np.log(p) - np.log1p(-p))
    return {key: result[key] for key in ("brier", "nll", "ece", "recall", "precision")}


def separation(y: Any, p: Any, score: Any) -> dict[str, Any]:
    y, p, score = np.asarray(y), np.asarray(p), np.asarray(score)
    errors = (p >= 0.5) != y
    auc = metrics(errors.astype(int), score)["roc_auc"]
    baseline = metrics(errors.astype(int), -np.abs(p - 0.5))["roc_auc"]
    quantiles = []
    edges = np.quantile(score, np.linspace(0, 1, 6))
    # Equal-value scores stay together; tied quantile bins may be empty.
    assignment = np.searchsorted(edges[1:-1], score, side="right")
    for index in range(5):
        mask = assignment == index
        quantiles.append(
            {
                "quantile": index + 1,
                "low": float(edges[index]),
                "high": float(edges[index + 1]),
                "n": int(mask.sum()),
                "error_rate": float(errors[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "errors_at_half": int(errors.sum()),
        "error_auroc": auc,
        "confidence_error_auroc": baseline,
        "gain": auc - baseline if auc is not None and baseline is not None else None,
        "error_quantiles": quantiles,
    }


def cascade_analysis(
    rows: dict[str, list[dict[str, Any]]], portfolio: tuple[str, ...], *, mode: str
) -> dict[str, Any]:
    """Replay aligned measured samples, excluding unvisited work from cost sums.

    mode=none/selective/always/confidence. Confidence routes ambiguous rows directly.
    Aggregate cost is a component estimate; actual runtime overhead is measured separately.
    """
    if mode not in {"none", "selective", "always", "confidence"}:
        raise ValueError("invalid diagnostic mode")
    first_rows = rows[portfolio[0]]
    y = np.array([r["label"] for r in first_rows])
    paths, final_p, final_positive, costs, uq_costs, extra, fallbacks = [], [], [], [], [], [], []
    by_tier: dict[str, Any] = {}
    for tier in portfolio:
        by_tier[tier] = {str(c): {"visited": 0, "requested": 0, "escalated": 0} for c in (0, 1)}
    for i, first in enumerate(first_rows):
        path, cost, uq_cost, passes, fallback = [], 0.0, 0.0, 0, False
        for j, tier in enumerate(portfolio):
            r = rows[tier][i]
            if r["candidate_id"] != first["candidate_id"] or r["label"] != first["label"]:
                raise ValueError("unaligned saved candidates")
            path.append(tier)
            cost += r["deterministic_ms"]
            group = by_tier[tier][str(r["label"])]
            group["visited"] += 1
            sample = mode == "always" or (mode == "selective" and r["ambiguous"])
            if sample:
                uq_cost += r["mc_ms"]
                passes += r["passes"]
            route = r["unresolved"] if sample else mode == "confidence" and r["ambiguous"]
            group["requested"] += int(route)
            if route and j < len(portfolio) - 1:
                group["escalated"] += 1
                continue
            fallback = bool(route)
            break
        paths.append(path)
        final_p.append(r["p"])
        final_positive.append(r["logit"] >= r["frozen_threshold_logit"])
        costs.append(cost + uq_cost)
        uq_costs.append(uq_cost)
        extra.append(passes)
        fallbacks.append(fallback)
    initial_error = np.array([(r["p"] >= 0.5) != r["label"] for r in first_rows])
    final_error = (np.array(final_p) >= 0.5) != y
    escalated = np.array([len(p) > 1 for p in paths])
    count = int(escalated.sum())
    correction = int(np.sum(initial_error & ~final_error))
    regression = int(np.sum(~initial_error & final_error))
    baseline_cost = float(np.mean([r["deterministic_ms"] for r in first_rows]))
    for groups in by_tier.values():
        for group in groups.values():
            n = group["visited"]
            group["escalation_rate"] = group["escalated"] / n if n else None
    return {
        "portfolio": list(portfolio),
        "mode": mode,
        "n": len(y),
        "escalated": count,
        "escalation_rate": count / len(y),
        "escalation_by_tier_and_class": by_tier,
        "initial_error_rate_escalated": float(initial_error[escalated].mean()) if count else None,
        "initial_error_rate_not_escalated": (
            float(initial_error[~escalated].mean()) if (~escalated).any() else None
        ),
        "final_error_rate_escalated": float(final_error[escalated].mean()) if count else None,
        "final_error_rate_not_escalated": (
            float(final_error[~escalated].mean()) if (~escalated).any() else None
        ),
        "corrections": correction,
        "regressions": regression,
        "correction_rate_initial_errors": correction / int(initial_error.sum())
        if initial_error.any()
        else None,
        "regression_rate_initial_correct": regression / int((~initial_error).sum())
        if (~initial_error).any()
        else None,
        "mean_uq_ms": float(np.mean(uq_costs)),
        "uq_ms_per_escalated_candidate": sum(uq_costs) / count if count else None,
        "mean_ms": float(np.mean(costs)),
        "p95_ms": float(np.quantile(costs, 0.95)),
        "component_throughput_per_s": 1000 / float(np.mean(costs)),
        "additional_passes_per_candidate": float(np.mean(extra)),
        "uq_to_initial_deterministic_cost": float(np.mean(uq_costs)) / baseline_cost,
        "conservative_fallbacks": sum(fallbacks),
        "recall_at_frozen_tier_thresholds": float(np.mean(np.array(final_positive)[y == 1])),
        "final_probability_metrics": probability_metrics(y, final_p),
        "paths": paths,
    }
