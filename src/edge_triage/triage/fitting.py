"""Validation-only deterministic byte minimization over final runtime evidence.

The caller supplies complete configuration outputs, never saved single-tier scores
masquerading as adaptive execution. Fitting does not run models or tune the routing.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from edge_triage.contracts import Contract, Count, ModelTier, NonNegative
from edge_triage.health import hash_file
from edge_triage.runtime.interfaces import FateContext, ThresholdDecision
from edge_triage.triage.artifacts import FateOperatingPoint, TierFateDefinition
from edge_triage.triage.policies import LogicalByteAccounting, PerTierRecallConstrainedPolicy


class ValidationFateRow(Contract):
    context: FateContext
    label: Literal[0, 1]
    original_bytes: Count
    end_to_end_ms: NonNegative


@dataclass(frozen=True)
class FitResult:
    tiers: tuple[TierFateDefinition, ...]
    metrics: dict[str, Any]
    searched_options: int
    objective: str


def _threshold(spec: TierFateDefinition) -> ThresholdDecision:
    return ThresholdDecision(
        p_real_threshold=0.5,
        artifact_id="VALIDATION_SEARCH_ONLY",
        artifact_sha256="0" * 64,
        checkpoint_id=spec.checkpoint_id,
        trusted_tier=spec.tier,
        discard_supported=spec.discard_supported,
        probability_mode=spec.probability_mode,
        calibration_artifact_id=spec.calibration_artifact_id,
        **spec.operating_point.model_dump(),
    )


def evaluate_fates(
    rows: Sequence[ValidationFateRow],
    tiers: tuple[TierFateDefinition, ...],
    accounting: LogicalByteAccounting,
    *,
    binary: bool = False,
) -> dict[str, Any]:
    policy = PerTierRecallConstrainedPolicy(accounting, binary)
    specs = {spec.tier: _threshold(spec) for spec in tiers}
    outcomes = []
    for row in rows:
        prediction = row.context.prediction
        threshold = specs.get(prediction.tier) if prediction else None
        outcomes.append(policy.decide_context(row.context, threshold, row.original_bytes))
    real = sum(row.label for row in rows)
    bogus = len(rows) - real
    false_discard = sum(
        r.label == 1 and o.action == "discard" for r, o in zip(rows, outcomes, strict=True)
    )
    transmitted = sum(o.output_bytes for o in outcomes)
    original = sum(o.input_bytes for o in outcomes)
    classifier_real = sum(
        r.label == 1 and o.p_real is not None and o.p_real >= 0.5 and not r.context.failed
        for r, o in zip(rows, outcomes, strict=True)
    )
    distribution: dict[str, dict[str, int]] = {}
    for row, outcome in zip(rows, outcomes, strict=True):
        tier = row.context.prediction.tier.value if row.context.prediction else "unavailable"
        key = f"{row.label}:{tier}"
        counts = distribution.setdefault(key, dict(discard=0, compress=0, transmit=0))
        counts[outcome.action] += 1
    return {
        "candidates": len(rows),
        "real_candidates": real,
        "survival_recall": (real - false_discard) / real if real else None,
        "classifier_recall": classifier_real / real if real else None,
        "classifier_score_unavailable": sum(o.p_real is None for o in outcomes),
        "real_transmit_recall": sum(
            r.label == 1 and o.action == "transmit" for r, o in zip(rows, outcomes, strict=True)
        )
        / real
        if real
        else None,
        "real_compress_recall": sum(
            r.label == 1 and o.action == "compress" for r, o in zip(rows, outcomes, strict=True)
        )
        / real
        if real
        else None,
        "bogus_discard_rate": sum(
            r.label == 0 and o.action == "discard" for r, o in zip(rows, outcomes, strict=True)
        )
        / bogus
        if bogus
        else None,
        "false_discard_count": false_discard,
        "false_discard_rate": false_discard / real if real else None,
        "bytes_original": original,
        "bytes_transmitted": transmitted,
        "byte_reduction": 1 - transmitted / original if original else None,
        "total_end_to_end_ms": sum(row.end_to_end_ms for row in rows),
        "cost_scope": "Observed runtime E2E including waits; not CPU compute attribution. "
        "Fate search time excluded; logical codec CPU=0; no network executed.",
        "fate_distribution_label_tier": distribution,
        "false_discard_review": [
            {
                "candidate_id": r.context.candidate_id,
                "tier": r.context.prediction.tier.value,
                "p_real": o.p_real,
            }
            for r, o in zip(rows, outcomes, strict=True)
            if r.label == 1 and o.action == "discard" and r.context.prediction is not None
        ],
    }


def _validation_rows(
    rows: Iterable[ValidationFateRow], partition: str, split_path: Path, split_sha256: str
) -> list[ValidationFateRow]:
    # Reject test access BEFORE consuming even one supplied prediction.
    if partition != "validation":
        raise ValueError("fate fitting is validation-only")
    if hash_file(split_path) != split_sha256:
        raise ValueError("split hash mismatch")
    with split_path.open(encoding="utf-8", newline="") as stream:
        expected = {
            r["candidate_id"]: int(r["label"])
            for r in csv.DictReader(stream)
            if r["split"] == "validation"
        }
    result = list(rows)
    ids = [r.context.candidate_id for r in result]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("fit requires each validation candidate exactly once, no train/test rows")
    if any(row.label != expected[row.context.candidate_id] for row in result):
        raise ValueError("validation labels disagree with frozen split")
    if not any(r.label == 1 for r in result):
        raise ValueError("survival recall undefined without Real validation candidates")
    return sorted(result, key=lambda r: r.context.candidate_id)


def fit_thresholds(
    rows: Iterable[ValidationFateRow],
    *,
    partition: str,
    split_path: Path,
    split_sha256: str,
    tiers: tuple[TierFateDefinition, ...],
    accounting: LogicalByteAccounting,
    target_survival_recall: float = 0.95,
    common: bool = False,
    monotonic: bool = True,
    transmit_grid: tuple[float, ...] = (0.5,),
    uncertainty_grid: tuple[float, ...] = (0.0,),
    binary: bool = False,
) -> FitResult:
    """Exact finite-grid optimization with a global integer false-discard budget.

    Dynamic programming retains the best prefix for each (false discards, final
    discard bound). Bounds use inclusive observed scores below .5, plus disabled.
    User-declared transmit/UQ grids stay fixed before fitting; no test access.
    """
    data = _validation_rows(rows, partition, split_path, split_sha256)
    if not math.isfinite(target_survival_recall) or not 0 < target_survival_recall <= 1:
        raise ValueError("target survival recall must be in (0,1]")
    if not tiers or [s.tier for s in tiers] != [
        t for t in ModelTier if t in {s.tier for s in tiers}
    ]:
        raise ValueError("unique ordered tier definitions required")
    if not transmit_grid or not uncertainty_grid:
        raise ValueError("nonempty predeclared search grids required")
    # A routing configuration may never reach some enabled tiers. Do not grant
    # discard authority to a tier with no compatible, resolved validation evidence.
    supported = set()
    probe = PerTierRecallConstrainedPolicy(accounting)
    for spec in tiers:
        for row in data:
            prediction = row.context.prediction
            if prediction is not None and prediction.tier == spec.tier:
                permissive = ThresholdDecision.model_validate(
                    {
                        **_threshold(spec).model_dump(),
                        "max_uncertainty_for_discard": max(uncertainty_grid),
                    }
                )
                decision = probe.decide_context(row.context, permissive, row.original_bytes)
                if not decision.reason_code.startswith("PRESERVE_"):
                    supported.add(spec.tier)
    tiers = tuple(
        TierFateDefinition.model_validate(
            {**s.model_dump(), "discard_supported": s.discard_supported and s.tier in supported}
        )
        for s in tiers
    )
    for transmit in transmit_grid:
        for std in uncertainty_grid:
            FateOperatingPoint(min_p_real_for_transmit=transmit, max_uncertainty_for_discard=std)
    spec_by_tier = {s.tier: s for s in tiers}
    scores = set()
    for row in data:
        prediction = row.context.prediction
        if prediction is None:
            continue
        if prediction.tier not in spec_by_tier:
            raise ValueError("validation output contains a disabled tier")
        spec = spec_by_tier[prediction.tier]
        score = (
            prediction.calibrated_p_real
            if spec.probability_mode == "calibrated"
            else prediction.p_real
        )
        if score is not None and score < 0.5:
            scores.add(score)
    bounds: list[float | None] = [None, *sorted(scores)]
    real = sum(r.label for r in data)
    budget = max(k for k in range(real + 1) if (real - k) / real >= target_survival_recall)

    def definitions(
        point: FateOperatingPoint, subset: tuple[TierFateDefinition, ...]
    ) -> tuple[TierFateDefinition, ...]:
        return tuple(
            TierFateDefinition.model_validate({**s.model_dump(), "operating_point": point})
            for s in subset
        )

    # Value: bytes, summed uncertainty cutoffs, deterministic point key, definitions.
    states: dict[tuple[int, int], tuple[Any, ...]] = {(0, 0): (0, 0.0, (), ())}
    groups = [tiers] if common else [(spec,) for spec in tiers]
    searched = 0
    for group in groups:
        group_tiers = {s.tier for s in group}
        subset = [
            r
            for r in data
            if r.context.prediction is not None and r.context.prediction.tier in group_tiers
        ]
        options: dict[tuple[int, int], tuple[Any, ...]] = {}
        # Resolve safety/identity once through the real policy, then vectorize only
        # the scalar threshold accounting. The chosen point is checked again below.
        permissive_specs = {
            s.tier: _threshold(
                TierFateDefinition.model_validate(
                    {
                        **s.model_dump(),
                        "operating_point": FateOperatingPoint(max_uncertainty_for_discard=0.5),
                    }
                )
            )
            for s in group
        }
        probes = [
            probe.decide_context(
                r.context, permissive_specs[r.context.prediction.tier], r.original_bytes
            )
            for r in subset
            if r.context.prediction
        ]
        eligible = np.array([not p.reason_code.startswith("PRESERVE_") for p in probes], dtype=bool)
        values = np.array([p.p_real if p.p_real is not None else 1.0 for p in probes])
        dispersion = np.array(
            [
                r.context.uncertainty.standard_deviation if r.context.uncertainty else 1.0
                for r in subset
            ]
        )
        truth = np.array([r.label == 1 for r in subset], dtype=bool)
        full = np.array([accounting.original(r.original_bytes) for r in subset], dtype=np.int64)
        compressed = np.array(
            [accounting.output("compress", r.original_bytes) for r in subset], dtype=np.int64
        )
        for bound_index, bound in enumerate(bounds):
            for transmit in sorted(set(transmit_grid)):
                for std in sorted(set(uncertainty_grid)):
                    point = FateOperatingPoint(
                        max_p_real_for_discard=bound,
                        min_p_real_for_transmit=transmit,
                        max_uncertainty_for_discard=std,
                    )
                    defs = definitions(point, group)
                    safe = eligible & (dispersion <= std)
                    discard = (
                        safe & (values <= bound)
                        if bound is not None
                        else np.zeros(len(subset), dtype=bool)
                    )
                    output = (
                        full if binary else np.where(safe & (values < transmit), compressed, full)
                    )
                    transmitted = int(np.where(discard, 0, output).sum())
                    searched += 1
                    false = int((discard & truth).sum())
                    if false > budget:
                        continue
                    key = (false, bound_index)
                    value = (
                        transmitted,
                        std * len(group),
                        ((bound_index, transmit, std),),
                        defs,
                    )
                    if key not in options or value[:3] < options[key][:3]:
                        options[key] = value
        next_states: dict[tuple[int, int], tuple[Any, ...]] = {}
        for (used, previous), prefix in states.items():
            for (false, bound_index), option in options.items():
                if used + false > budget or (monotonic and bound_index < previous):
                    continue
                key = (used + false, bound_index)
                value = (
                    prefix[0] + option[0],
                    prefix[1] + option[1],
                    prefix[2] + option[2],
                    prefix[3] + option[3],
                )
                if key not in next_states or value[:3] < next_states[key][:3]:
                    next_states[key] = value
        states = next_states
    # Equal bytes: fewer lost Real, lower uncertainty cutoff, then lower bounds.
    # Observed compute is identical for all fate thresholds on this fixed run.
    selected_key, selected = min(
        states.items(), key=lambda kv: (kv[1][0], kv[0][0], kv[1][1], kv[1][2])
    )
    fitted = selected[3]
    metrics = evaluate_fates(data, fitted, accounting, binary=binary)
    assert metrics["false_discard_count"] == selected_key[0]
    # Rows without predictions are preserved, independently of the optimized groups.
    unpredicted_bytes = sum(
        accounting.original(r.original_bytes) for r in data if r.context.prediction is None
    )
    assert metrics["bytes_transmitted"] == selected[0] + unpredicted_bytes
    assert metrics["survival_recall"] >= target_survival_recall
    return FitResult(
        fitted,
        metrics,
        searched,
        "Minimize logical transmitted bytes; ties: fewer real discards, lower "
        "summed uncertainty cutoffs, then lexicographic bounds. Fixed-run compute "
        "is invariant. Conditional validation fit; no held-out guarantee.",
    )


def bootstrap_fates(
    rows: Sequence[ValidationFateRow],
    tiers: tuple[TierFateDefinition, ...],
    accounting: LogicalByteAccounting,
    *,
    repeats: int = 1000,
    seed: int = 20260924,
    binary: bool = False,
) -> dict[str, Any]:
    if not rows or repeats < 1:
        raise ValueError("nonempty data and positive bootstrap repetitions required")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        name: [] for name in ("survival_recall", "classifier_recall", "byte_reduction")
    }
    policy = PerTierRecallConstrainedPolicy(accounting, binary)
    specs = {s.tier: _threshold(s) for s in tiers}
    outcomes = [
        policy.decide_context(
            r.context,
            specs.get(r.context.prediction.tier) if r.context.prediction else None,
            r.original_bytes,
        )
        for r in rows
    ]
    real = np.array([r.label == 1 for r in rows])
    surviving = real & np.array([o.action != "discard" for o in outcomes])
    classified = real & np.array(
        [
            o.p_real is not None and o.p_real >= 0.5 and not r.context.failed
            for r, o in zip(rows, outcomes, strict=True)
        ]
    )
    incoming = np.array([o.input_bytes for o in outcomes], dtype=np.int64)
    outgoing = np.array([o.output_bytes for o in outcomes], dtype=np.int64)
    for _ in range(repeats):
        indices = rng.integers(0, len(rows), len(rows))
        denominator = int(real[indices].sum())
        if denominator:
            samples["survival_recall"].append(float(surviving[indices].sum() / denominator))
            samples["classifier_recall"].append(float(classified[indices].sum() / denominator))
        original = int(incoming[indices].sum())
        if original:
            samples["byte_reduction"].append(float(1 - outgoing[indices].sum() / original))
    return {
        "method": "IID percentile 95%; fixed fitted thresholds; conditional fit-set diagnostics",
        "seed": seed,
        "repeats": repeats,
        "intervals": {
            name: {
                "lower": float(np.quantile(values, 0.025)) if values else None,
                "upper": float(np.quantile(values, 0.975)) if values else None,
                "defined_replicates": len(values),
            }
            for name, values in samples.items()
        },
    }
