import itertools
import math

import pytest
from pydantic import ValidationError

from edge_triage.contracts import ModelTier, PredictionResult, UncertaintyResult
from edge_triage.data.artifacts import canonical, write_immutable
from edge_triage.health import hash_file
from edge_triage.runtime.interfaces import FateContext, ThresholdDecision
from edge_triage.triage.artifacts import (
    CommonThresholdPolicy,
    FateOperatingPoint,
    FateThresholdArtifact,
    PerTierThresholdPolicy,
    TierFateDefinition,
)
from edge_triage.triage.fitting import (
    ValidationFateRow,
    bootstrap_fates,
    evaluate_fates,
    fit_thresholds,
)
from edge_triage.triage.policies import (
    BinaryDiscardTransmitPolicy,
    CommonThresholdThreeWayPolicy,
    LogicalByteAccounting,
    PerTierRecallConstrainedPolicy,
    TransmitAllPolicy,
)


def evidence(p=0.1, tier=ModelTier.TINY, candidate_id="c", **changes):
    prediction = PredictionResult(
        candidate_id=candidate_id,
        tier=tier,
        checkpoint_id=tier.value,
        logit=math.log(p / (1 - p)),
        p_real=p,
        inference_ms=1,
    )
    uq = UncertaintyResult(
        candidate_id=candidate_id,
        tier=tier,
        method="none",
        passes=1,
        mean_p_real=p,
        variance=0,
        predictive_entropy=0,
        duration_ms=0,
    )
    return FateContext.model_validate(
        dict(candidate_id=candidate_id, prediction=prediction, uncertainty=uq) | changes
    )


def threshold(**changes):
    return ThresholdDecision.model_validate(
        dict(
            p_real_threshold=0.5,
            artifact_id="test",
            artifact_sha256="a" * 64,
            checkpoint_id="tiny",
            trusted_tier="tiny",
            discard_supported=True,
            probability_mode="uncalibrated",
            max_p_real_for_discard=0.2,
            min_p_real_for_transmit=0.8,
            max_uncertainty_for_discard=0.1,
        )
        | changes
    )


@pytest.mark.parametrize(
    "policy",
    [
        PerTierRecallConstrainedPolicy(),
        CommonThresholdThreeWayPolicy(),
        BinaryDiscardTransmitPolicy(),
    ],
)
def test_probability_direction_and_inclusive_boundaries(policy):
    assert policy.decide_context(evidence(0.2), threshold(), 100).action == "discard"
    assert policy.decide_context(evidence(0.8), threshold(), 100).action == "transmit"
    assert policy.decide_context(evidence(0.99), threshold(), 100).action == "transmit"
    assert policy.decide_context(evidence(0.4), threshold(), 100).action == (
        "transmit" if policy.binary else "compress"
    )
    # No evidence is inferred from the old prediction-only interface.
    assert policy.decide(evidence().prediction, threshold(), 100).action == "transmit"
    with pytest.raises(ValidationError):
        threshold(max_p_real_for_discard=0.8)


@pytest.mark.parametrize(
    "case",
    [
        "missing_prediction",
        "missing_uq",
        "missing_calibration",
        "failed",
        "terminal",
        "ambiguous",
        "unresolved",
        "high_std",
        "wrong_checkpoint",
        "wrong_uq",
        "wrong_mode",
        "missing_threshold",
    ],
)
def test_conservative_missing_failure_and_ambiguity(case):
    context, point = evidence(), threshold()
    raw = context.model_dump()
    if case == "missing_prediction":
        raw["prediction"] = None
    elif case == "missing_uq":
        raw["uncertainty"] = None
    elif case == "missing_calibration":
        point = threshold(probability_mode="calibrated", calibration_artifact_id="cal")
    elif case == "failed":
        raw["failed"] = True
    elif case == "terminal":
        raw["conservative_fallback"] = True
    elif case in ("ambiguous", "unresolved"):
        raw["uncertainty"][case] = True
    elif case == "high_std":
        raw["uncertainty"]["standard_deviation"] = 0.2
    elif case == "wrong_checkpoint":
        point = threshold(checkpoint_id="different")
    elif case == "wrong_uq":
        raw["uncertainty"]["candidate_id"] = "different"
    elif case == "wrong_mode":
        raw["prediction"].update(calibrated_p_real=0.1, calibration_artifact_id="cal")
    elif case == "missing_threshold":
        point = None
    result = PerTierRecallConstrainedPolicy().decide_context(
        FateContext.model_validate(raw), point, 120000
    )
    assert result.action == "transmit" and result.output_bytes == 120000
    assert result.reason_code.startswith("PRESERVE_")


def test_explicit_raw_mode_and_calibrated_score():
    raw = evidence(0.1).model_dump()
    raw["prediction"].update(calibrated_p_real=0.99, calibration_artifact_id="cal")
    point = threshold(probability_mode="calibrated", calibration_artifact_id="cal")
    result = PerTierRecallConstrainedPolicy().decide_context(
        FateContext.model_validate(raw), point, 10
    )
    assert result.action == "transmit" and result.p_real == 0.99


def test_unvalidated_or_wrong_tier_cannot_discard():
    for point in (threshold(discard_supported=False), threshold(trusted_tier="large")):
        result = PerTierRecallConstrainedPolicy().decide_context(evidence(), point, 100)
        assert result.action == "transmit"
        assert result.reason_code in {"PRESERVE_UNVALIDATED_TIER", "PRESERVE_TIER_MISMATCH"}


def test_byte_accounting_and_survival_are_exact():
    accounting = LogicalByteAccounting(compression_fraction=0.25, metadata_bytes=7)
    assert [accounting.output(a, 101) for a in ("discard", "compress", "transmit")] == [0, 33, 108]
    with pytest.raises(ValueError):
        accounting.output("transmit", -1)
    rows = [
        ValidationFateRow(
            context=evidence(p, candidate_id=f"c{i}"),
            label=label,
            original_bytes=100,
            end_to_end_ms=2,
        )
        for i, (p, label) in enumerate([(0.1, 1), (0.4, 1), (0.9, 1), (0.1, 0)])
    ]
    spec = TierFateDefinition(
        tier="tiny",
        checkpoint_id="tiny",
        probability_mode="uncalibrated",
        operating_point=FateOperatingPoint(max_p_real_for_discard=0.2, min_p_real_for_transmit=0.8),
    )
    metrics = evaluate_fates(rows, (spec,), accounting)
    assert metrics["survival_recall"] == 2 / 3
    assert metrics["classifier_recall"] == 1 / 3
    assert metrics["real_transmit_recall"] == metrics["real_compress_recall"] == 1 / 3
    assert metrics["false_discard_count"] == 1 and metrics["bogus_discard_rate"] == 1
    assert metrics["bytes_original"] == 428 and metrics["bytes_transmitted"] == 139
    assert metrics["false_discard_review"][0]["candidate_id"] == "c0"
    baseline = TransmitAllPolicy(accounting).decide_context(evidence(), threshold(), 100)
    assert baseline.input_bytes == baseline.output_bytes == 107
    bootstrap = bootstrap_fates(rows, (spec,), accounting, repeats=20)
    assert bootstrap == bootstrap_fates(rows, (spec,), accounting, repeats=20)


def artifact(tiers, **changes):
    return FateThresholdArtifact.model_validate(
        dict(
            artifact_id="test-frozen",
            threshold_policy="per_tier",
            system_sha256="a" * 64,
            family_sha256="b" * 64,
            split_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            calibration_index_sha256="e" * 64,
            validation_records_sha256="f" * 64,
            target_survival_recall=0.95,
            validation_survival_recall=1,
            objective="test fixture only",
            validation_metrics={},
            tiers=tiers,
        )
        | changes
    )


def test_common_per_tier_monotonic_frozen_artifact(tmp_path):
    specs = tuple(
        TierFateDefinition(
            tier=tier,
            checkpoint_id=tier.value,
            probability_mode="uncalibrated",
            operating_point=FateOperatingPoint(max_p_real_for_discard=0.2),
        )
        for tier in ModelTier
    )
    point = artifact(specs, threshold_policy="common")
    common, per_tier = CommonThresholdPolicy(point), PerTierThresholdPolicy(point)
    assert all(common.resolve(t) == per_tier.resolve(t) for t in ModelTier)
    path = tmp_path / "frozen.json"
    common.persist(path)
    assert hash_file(path) == common.digest
    with pytest.raises(ValidationError):
        point.tiers[0].operating_point.max_p_real_for_discard = 0.3
    with pytest.raises(FileExistsError):
        write_immutable(path, canonical({"changed": True}))
    changed = [s.model_dump() for s in specs]
    changed[0]["operating_point"]["max_p_real_for_discard"] = 0.3
    with pytest.raises(ValidationError, match="weaker"):
        artifact(changed)
    with pytest.raises(ValidationError, match="below target"):
        artifact(specs, validation_survival_recall=0.9)


def fit_fixture(tmp_path):
    rows = [
        ValidationFateRow(
            context=evidence(p, tier, f"c{i}"), label=label, original_bytes=100, end_to_end_ms=2
        )
        for i, (tier, p, label) in enumerate(
            [
                (ModelTier.TINY, 0.1, 0),
                (ModelTier.TINY, 0.2, 1),
                (ModelTier.TINY, 0.9, 1),
                (ModelTier.LARGE, 0.3, 0),
                (ModelTier.LARGE, 0.4, 1),
                (ModelTier.LARGE, 0.95, 1),
            ]
        )
    ]
    split = tmp_path / "split.csv"
    split.write_text(
        "candidate_id,label,split\n"
        + "".join(f"{r.context.candidate_id},{r.label},validation\n" for r in rows)
        + "test-id,1,test\n"
    )
    specs = tuple(
        TierFateDefinition(
            tier=t,
            checkpoint_id=t.value,
            probability_mode="uncalibrated",
            operating_point=FateOperatingPoint(),
        )
        for t in (ModelTier.TINY, ModelTier.LARGE)
    )
    args = dict(
        partition="validation",
        split_path=split,
        split_sha256=hash_file(split),
        tiers=specs,
        accounting=LogicalByteAccounting(),
        target_survival_recall=0.75,
    )
    return rows, args


@pytest.mark.parametrize("monotonic,common", [(True, True), (True, False), (False, False)])
def test_fitter_matches_exhaustive_global_optimum(tmp_path, monotonic, common):
    rows, args = fit_fixture(tmp_path)
    result = fit_thresholds(rows, **args, monotonic=monotonic, common=common)
    best = None
    for a, b in itertools.product([None, 0.1, 0.2, 0.3, 0.4], repeat=2):
        if (
            common
            and a != b
            or monotonic
            and (a if a is not None else -1) > (b if b is not None else -1)
        ):
            continue
        specs = tuple(
            TierFateDefinition.model_validate(
                s.model_dump() | {"operating_point": FateOperatingPoint(max_p_real_for_discard=d)}
            )
            for s, d in zip(args["tiers"], [a, b], strict=True)
        )
        metrics = evaluate_fates(rows, specs, args["accounting"])
        if metrics["survival_recall"] >= 0.75:
            rank = (metrics["bytes_transmitted"], metrics["false_discard_count"])
            best = min(best, rank) if best else rank
    assert (result.metrics["bytes_transmitted"], result.metrics["false_discard_count"]) == best
    assert fit_thresholds(reversed(rows), **args, monotonic=monotonic, common=common) == result


def test_fitting_rejects_test_before_access_and_requires_exact_validation(tmp_path):
    rows, args = fit_fixture(tmp_path)

    def forbidden():
        raise AssertionError("test rows must not be touched")
        yield

    with pytest.raises(ValueError, match="validation-only"):
        fit_thresholds(forbidden(), **(args | {"partition": "test"}))
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="exactly once"):
            fit_thresholds(bad, **args)
    with pytest.raises(ValueError, match="hash"):
        fit_thresholds(rows, **(args | {"split_sha256": "0" * 64}))


def test_fitter_disables_unobserved_portfolio_tier(tmp_path):
    rows, args = fit_fixture(tmp_path)
    medium = TierFateDefinition(
        tier="medium",
        checkpoint_id="medium",
        probability_mode="uncalibrated",
        operating_point=FateOperatingPoint(),
    )
    args["tiers"] = (args["tiers"][0], medium, args["tiers"][1])
    result = fit_thresholds(rows, **args)
    assert result.tiers[1].discard_supported is False
    assert result.tiers[0].discard_supported and result.tiers[2].discard_supported
