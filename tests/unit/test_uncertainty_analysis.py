import pytest

from edge_triage.uncertainty.analysis import cascade_analysis, fit_interval, fit_std, separation


def test_validation_only_fit_inclusive_ties_and_invalid_values():
    lo, hi = fit_interval([0.1, 0.4, 0.4, 0.6, 0.9], 0.2, partition="validation")
    assert lo <= 0.4 <= hi and lo <= 0.6 <= hi
    assert fit_std([0, 0.1, 0.2, 0.3, 0.4], 0.8, partition="validation") == pytest.approx(0.32)
    for partition in ("test", "train"):
        with pytest.raises(ValueError):
            fit_interval([0.5], 0.2, partition=partition)
        with pytest.raises(ValueError):
            fit_std([0.1], 0.8, partition=partition)
    for values in ([], [float("nan")], [1.1]):
        with pytest.raises(ValueError):
            fit_interval(values, 0.2, partition="validation")


def test_error_auc_and_undefined_no_errors():
    result = separation([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.2], [0, 0, 1, 1])
    assert result["error_auroc"] == 1
    assert sum(q["n"] for q in result["error_quantiles"]) == 4
    assert separation([0, 1], [0.1, 0.9], [1, 1])["error_auroc"] is None


def test_cascade_metrics_known_corrections_regressions_and_visited_cost():
    rows = {}
    for tier, probabilities in [("tiny", [0.4, 0.8, 0.7, 0.1]), ("large", [0.9, 0.4, 0.1, 0.9])]:
        rows[tier] = [
            {
                "candidate_id": str(i),
                "label": y,
                "p": p,
                "logit": p - 0.5,
                "frozen_threshold_logit": 0,
                "deterministic_ms": 1,
                "mc_ms": 2,
                "passes": 5,
                "ambiguous": i < 2,
                "unresolved": True,
            }
            for i, (p, y) in enumerate(zip(probabilities, [1, 1, 0, 0]))
        ]
    report = cascade_analysis(rows, ("tiny", "large"), mode="selective")
    assert report["corrections"] == report["regressions"] == 1
    assert report["escalated"] == 2 and report["additional_passes_per_candidate"] == 5
    assert report["mean_uq_ms"] == 2 and report["mean_ms"] == 3.5
    assert report["recall_at_frozen_tier_thresholds"] == 0.5
    confidence = cascade_analysis(rows, ("tiny", "large"), mode="confidence")
    assert confidence["additional_passes_per_candidate"] == 0
    assert confidence["mean_ms"] == 1.5
    rows["large"][0]["candidate_id"] = "misaligned"
    with pytest.raises(ValueError):
        cascade_analysis(rows, ("tiny", "large"), mode="selective")
